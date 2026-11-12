# Copyright (c) 2025-2026 Buf Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from textwrap import dedent
from typing import TYPE_CHECKING, Any, TypeAlias

from typing_extensions import assert_never

from ._descriptors import (
    DescEnum,
    DescField,
    DescFieldValueEnum,
    DescFieldValueList,
    DescFieldValueMap,
    DescFieldValueMessage,
    DescFieldValueScalar,
    DescMessage,
    ScalarType,
)
from ._wkt_registry import (
    WktAny,
    WktDuration,
    WktFieldMask,
    WktFileDescriptorSet,
    WktListValue,
    WktMatch,
    WktStruct,
    WktTimestamp,
    WktValue,
    WktWrapper,
    match_wkt,
)

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema

    from ._message import Message

JSON: TypeAlias = (
    Mapping[str, "JSON"] | Sequence["JSON"] | str | int | float | bool | None
)


@dataclass(frozen=True, kw_only=True, slots=True)
class JsonSchemaField:
    description: str = ""
    # A field's label. Set to the protobuf field name; without it pydantic
    # derives one from the JSON property name (e.g. "Explicitfield").
    title: str = ""
    ref: str = ""
    type: str = ""
    format: str = ""
    pattern: str = ""
    content_encoding: str = ""
    enum: Sequence[str] = ()
    # Sample values, surfaced by tools like Swagger UI as the placeholder. Used
    # for scalars whose JSON form is a string but not free text (64-bit integers
    # and bytes), where the tool's generic "string" placeholder is misleading.
    examples: Sequence[JSON] = ()
    properties: Mapping[str, JsonSchemaField] | None = None
    # We omit additionalProperties with the value type for maps, otherwise
    # since ProtoJSON is permissive we leave the default (true).
    additional_properties: JsonSchemaField | None = None
    items: JsonSchemaField | None = None

    def to_json(self) -> dict[str, JSON]:
        res: dict[str, JSON] = {}
        if self.description:
            res["description"] = self.description
        if self.title:
            res["title"] = self.title
        if self.ref:
            res["$ref"] = self.ref
        if self.type:
            res["type"] = self.type
        if self.format:
            res["format"] = self.format
        if self.pattern:
            res["pattern"] = self.pattern
        if self.content_encoding:
            res["contentEncoding"] = self.content_encoding
        if self.enum:
            res["enum"] = list(self.enum)
        if self.examples:
            res["examples"] = list(self.examples)
        if self.properties:
            res["properties"] = {k: v.to_json() for k, v in self.properties.items()}
        if self.additional_properties:
            res["additionalProperties"] = self.additional_properties.to_json()
        if self.items:
            res["items"] = self.items.to_json()
        return res


def _generate_wkt_schema(wkt: WktMatch) -> JsonSchemaField | None:
    match wkt:
        case WktTimestamp():
            return JsonSchemaField(type="string", format="date-time")
        case WktDuration():
            return JsonSchemaField(type="string", format="duration")
        case WktFieldMask():
            return JsonSchemaField(type="string")
        case WktAny():
            return JsonSchemaField(
                type="object", properties={"@type": JsonSchemaField(type="string")}
            )
        case WktStruct():
            return JsonSchemaField(type="object")
        case WktListValue():
            return JsonSchemaField(type="array")
        case WktValue():
            # Any JSON value.
            return JsonSchemaField()
        case WktWrapper():
            return _scalar_field(wkt.value.scalar)
        case WktFileDescriptorSet():
            return None
        case _:
            assert_never(wkt)


def _scalar_field(scalar: ScalarType, description: str = "") -> JsonSchemaField:
    # 64-bit integers and bytes serialize as JSON strings rather than free text,
    # so constrain them and give a sample value to avoid placeholders like "string" for
    # them.
    match scalar:
        case ScalarType.INT64 | ScalarType.SINT64 | ScalarType.SFIXED64:
            return JsonSchemaField(
                type="string",
                pattern="^-?[0-9]+$",
                examples=("0",),
                description=description,
            )
        case ScalarType.UINT64 | ScalarType.FIXED64:
            return JsonSchemaField(
                type="string",
                pattern="^[0-9]+$",
                examples=("0",),
                description=description,
            )
        case ScalarType.BYTES:
            return JsonSchemaField(
                type="string",
                content_encoding="base64",
                examples=("",),
                description=description,
            )
        case _:
            return JsonSchemaField(type=_scalar_type(scalar), description=description)


def _scalar_type(s: ScalarType) -> str:
    match s:
        case ScalarType.BOOL:
            return "boolean"
        case (
            ScalarType.INT32
            | ScalarType.UINT32
            | ScalarType.FIXED32
            | ScalarType.SINT32
            | ScalarType.SFIXED32
        ):
            return "integer"
        case (
            ScalarType.INT64
            | ScalarType.UINT64
            | ScalarType.FIXED64
            | ScalarType.SINT64
            | ScalarType.SFIXED64
        ):
            return "string"
        case ScalarType.FLOAT | ScalarType.DOUBLE:
            return "number"
        case ScalarType.STRING | ScalarType.BYTES:
            return "string"
        case _:
            assert_never(s)


def _generate_docstring(desc: DescMessage | DescField | DescEnum) -> str:
    """Recover a description from the generated Python docstrings.

    The code generator already renders proto comments into class docstrings, so
    we parse them back here instead of relying on source code info being retained
    on the runtime descriptors. A message/enum description is the leading text of
    its class docstring; a field description lives under the "Attributes:" section
    keyed by the proto field name. Both end at the ```proto code block the
    generator appends.
    """
    if isinstance(desc, DescField):
        return _docstring_field(desc.parent.type, desc.name)
    return _text_before_proto_fence(_cleaned_docstring(desc.type))


def _cleaned_docstring(cls: type) -> list[str]:
    # cls.__doc__ (not inspect.getdoc) so we don't inherit a base class's
    # docstring; cleandoc normalizes the indentation added by nesting.
    doc = cls.__doc__
    return inspect.cleandoc(doc).splitlines() if doc else []


def _docstring_field(message_cls: type, field_name: str) -> str:
    lines = _cleaned_docstring(message_cls)
    header = f"    {field_name}:"
    if header not in lines:
        return ""
    return _text_before_proto_fence(lines[lines.index(header) + 1 :])


def _text_before_proto_fence(lines: list[str]) -> str:
    body: list[str] = []
    for line in lines:
        if line.lstrip().startswith("```proto"):
            break
        body.append(line)
    return dedent("\n".join(body)).strip()


def _identity(value: Any) -> Any:
    return value


# Message type names currently being built, to break reference cycles. Pydantic
# only guards recursion for types it generates itself; when we drive nested type
# generation from this hook we have to detect cycles ourselves.
_BUILDING_TYPES: ContextVar[frozenset[str]] = ContextVar(
    "_protobuf_jsonschema_building", default=frozenset()
)


def build_pydantic_core_schema(
    cls: type[Message], handler: GetCoreSchemaHandler
) -> CoreSchema:
    """Build a pydantic core schema for a message class.

    Validation and serialization are delegated to our own JSON
    functions. The schema's structure is handed to pydantic to produce a
    JSON schema, notably what is rendered by FastAPI in its docs page.
    """
    from pydantic_core import core_schema  # noqa: PLC0415

    from ._from_json import message_from_json_value  # noqa: PLC0415
    from ._to_json import message_to_json_value  # noqa: PLC0415

    desc = cls._desc

    def validate(value: Any, _handler: Any) -> Any:
        if isinstance(value, cls):
            return value
        return message_from_json_value(cls, value, ignore_unknown_fields=True)

    def serialize(value: Any, info: Any) -> Any:
        if info.mode == "json":
            return message_to_json_value(value)
        return value

    serialization = core_schema.plain_serializer_function_ser_schema(
        serialize, info_arg=True
    )

    def leaf(field: JsonSchemaField) -> CoreSchema:
        # An inert carrier whose JSON schema is exactly our generated body. The
        # value is never validated through it (the wrap validator ignores the
        # inner schema).
        body = field.to_json()
        return core_schema.any_schema(
            metadata={"pydantic_js_functions": [lambda _s, _h: dict(body)]}
        )

    def enum_ref(enum: DescEnum) -> CoreSchema:
        # A schema-only definition so the enum becomes a shared, referenced $def.
        body = JsonSchemaField(
            type="string",
            enum=[value.name for value in enum.values],
            description=_generate_docstring(enum),
        ).to_json()
        return core_schema.no_info_plain_validator_function(
            _identity,
            ref=enum.type_name,
            metadata={"pydantic_js_functions": [lambda _s, _h: dict(body)]},
        )

    def message_ref(message: DescMessage) -> CoreSchema:
        # Pydantic doesn't automatically handle recusion when building a custom schema
        # like we do here, so we detect ourselves to get the reference.
        if message.type_name in _BUILDING_TYPES.get():
            return core_schema.definition_reference_schema(message.type_name)
        return handler.generate_schema(message.type)

    def singular(value: DescMessage | DescEnum | ScalarType) -> CoreSchema:
        if isinstance(value, DescMessage):
            return message_ref(value)
        if isinstance(value, DescEnum):
            return enum_ref(value)
        return leaf(_scalar_field(value))

    def field_core(field: DescField) -> CoreSchema:
        description = _generate_docstring(field)
        # Scalars carry title/description in their own JSON body. List and dict
        # schemas are generated by pydantic, so their title/description are merged
        # in through an update hook instead. Message/enum fields are bare $refs
        # and can carry neither (a sibling would force the ref to be inlined).
        field_updates: dict[str, JSON] = {"title": field.name}
        if description:
            field_updates["description"] = description
        field_metadata = {"pydantic_js_updates": field_updates}
        match value := field.value:
            case DescFieldValueScalar():
                scalar = _scalar_field(value.scalar, description)
                return leaf(replace(scalar, title=field.name))
            case DescFieldValueMessage():
                return message_ref(value.message)
            case DescFieldValueEnum():
                return enum_ref(value.enum)
            case DescFieldValueList():
                return core_schema.list_schema(
                    singular(value.element), metadata=field_metadata
                )
            case DescFieldValueMap():
                return core_schema.dict_schema(
                    core_schema.str_schema(),
                    singular(value.value),
                    metadata=field_metadata,
                )
            case _:
                assert_never(value)

    # Well-known types have a custom JSON form and no message-typed fields to
    # recurse into, so replace their whole schema with our body.
    if (wkt := match_wkt(desc)) is not None and (
        wkt_field := _generate_wkt_schema(wkt)
    ) is not None:
        wkt_body = wkt_field.to_json()
        return core_schema.no_info_wrap_validator_function(
            validate,
            core_schema.any_schema(),
            ref=desc.type_name,
            serialization=serialization,
            metadata={"pydantic_js_functions": [lambda _s, _h: dict(wkt_body)]},
        )

    token = _BUILDING_TYPES.set(_BUILDING_TYPES.get() | {desc.type_name})
    try:
        fields = {
            field.json_name: core_schema.typed_dict_field(
                field_core(field), required=False
            )
            for field in desc.fields
        }
    finally:
        _BUILDING_TYPES.reset(token)

    message_description = _generate_docstring(desc)
    metadata = (
        {"pydantic_js_updates": {"description": message_description}}
        if message_description
        else None
    )
    inner = core_schema.typed_dict_schema(
        fields, extra_behavior="ignore", total=False, metadata=metadata
    )
    return core_schema.no_info_wrap_validator_function(
        validate, inner, ref=desc.type_name, serialization=serialization
    )
