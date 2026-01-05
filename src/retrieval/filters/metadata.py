"""
Enterprise RAG System - Metadata Filters

Provides a flexible filtering system for metadata-based retrieval filtering.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union


class FilterOperator(str, Enum):
    """Filter comparison operators."""
    EQ = "eq"           # Equal
    NE = "ne"           # Not equal
    GT = "gt"           # Greater than
    GTE = "gte"         # Greater than or equal
    LT = "lt"           # Less than
    LTE = "lte"         # Less than or equal
    IN = "in"           # In list
    NIN = "nin"         # Not in list
    CONTAINS = "contains"  # String contains
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    EXISTS = "exists"   # Field exists
    RANGE = "range"     # Between two values


@dataclass
class FilterCondition:
    """A single filter condition."""
    field: str
    operator: FilterOperator
    value: Any

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
        }

    def matches(self, metadata: dict[str, Any]) -> bool:
        """Check if metadata matches this condition."""
        field_value = metadata.get(self.field)

        if self.operator == FilterOperator.EXISTS:
            return (self.field in metadata) == self.value

        if field_value is None:
            return False

        if self.operator == FilterOperator.EQ:
            return field_value == self.value
        elif self.operator == FilterOperator.NE:
            return field_value != self.value
        elif self.operator == FilterOperator.GT:
            return field_value > self.value
        elif self.operator == FilterOperator.GTE:
            return field_value >= self.value
        elif self.operator == FilterOperator.LT:
            return field_value < self.value
        elif self.operator == FilterOperator.LTE:
            return field_value <= self.value
        elif self.operator == FilterOperator.IN:
            return field_value in self.value
        elif self.operator == FilterOperator.NIN:
            return field_value not in self.value
        elif self.operator == FilterOperator.CONTAINS:
            return str(self.value).lower() in str(field_value).lower()
        elif self.operator == FilterOperator.STARTS_WITH:
            return str(field_value).lower().startswith(str(self.value).lower())
        elif self.operator == FilterOperator.ENDS_WITH:
            return str(field_value).lower().endswith(str(self.value).lower())
        elif self.operator == FilterOperator.RANGE:
            if isinstance(self.value, (list, tuple)) and len(self.value) == 2:
                return self.value[0] <= field_value <= self.value[1]
            return False

        return False


@dataclass
class MetadataFilter:
    """
    Composite metadata filter with AND/OR logic.

    Example:
        filter = MetadataFilter()
        filter.add_condition("source", FilterOperator.EQ, "confluence")
        filter.add_condition("date", FilterOperator.GTE, "2024-01-01")
    """
    conditions: list[FilterCondition] = field(default_factory=list)
    logic: str = "AND"  # "AND" or "OR"

    def add_condition(
        self,
        field: str,
        operator: Union[FilterOperator, str],
        value: Any,
    ) -> "MetadataFilter":
        """Add a filter condition."""
        if isinstance(operator, str):
            operator = FilterOperator(operator)

        self.conditions.append(FilterCondition(
            field=field,
            operator=operator,
            value=value,
        ))
        return self

    def add_eq(self, field: str, value: Any) -> "MetadataFilter":
        """Add equality condition."""
        return self.add_condition(field, FilterOperator.EQ, value)

    def add_in(self, field: str, values: list) -> "MetadataFilter":
        """Add 'in list' condition."""
        return self.add_condition(field, FilterOperator.IN, values)

    def add_range(
        self,
        field: str,
        min_value: Any,
        max_value: Any,
    ) -> "MetadataFilter":
        """Add range condition."""
        return self.add_condition(field, FilterOperator.RANGE, [min_value, max_value])

    def add_date_range(
        self,
        field: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> "MetadataFilter":
        """Add date range condition."""
        if start_date:
            self.add_condition(field, FilterOperator.GTE, start_date.isoformat())
        if end_date:
            self.add_condition(field, FilterOperator.LTE, end_date.isoformat())
        return self

    def matches(self, metadata: dict[str, Any]) -> bool:
        """Check if metadata matches the filter."""
        if not self.conditions:
            return True

        results = [c.matches(metadata) for c in self.conditions]

        if self.logic == "AND":
            return all(results)
        else:  # OR
            return any(results)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "logic": self.logic,
            "conditions": [c.to_dict() for c in self.conditions],
        }

    def to_qdrant_filter(self) -> dict[str, Any]:
        """Convert to Qdrant filter format."""
        if not self.conditions:
            return {}

        qdrant_conditions = []

        for cond in self.conditions:
            if cond.operator == FilterOperator.EQ:
                qdrant_conditions.append({
                    "key": cond.field,
                    "match": {"value": cond.value},
                })
            elif cond.operator == FilterOperator.IN:
                qdrant_conditions.append({
                    "key": cond.field,
                    "match": {"any": cond.value},
                })
            elif cond.operator in (FilterOperator.GT, FilterOperator.GTE,
                                    FilterOperator.LT, FilterOperator.LTE):
                range_filter = {}
                if cond.operator == FilterOperator.GT:
                    range_filter["gt"] = cond.value
                elif cond.operator == FilterOperator.GTE:
                    range_filter["gte"] = cond.value
                elif cond.operator == FilterOperator.LT:
                    range_filter["lt"] = cond.value
                elif cond.operator == FilterOperator.LTE:
                    range_filter["lte"] = cond.value

                qdrant_conditions.append({
                    "key": cond.field,
                    "range": range_filter,
                })
            elif cond.operator == FilterOperator.RANGE:
                if isinstance(cond.value, (list, tuple)) and len(cond.value) == 2:
                    qdrant_conditions.append({
                        "key": cond.field,
                        "range": {"gte": cond.value[0], "lte": cond.value[1]},
                    })

        if self.logic == "AND":
            return {"must": qdrant_conditions}
        else:
            return {"should": qdrant_conditions}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetadataFilter":
        """Create filter from dictionary."""
        filter_obj = cls(logic=data.get("logic", "AND"))

        for cond_data in data.get("conditions", []):
            filter_obj.add_condition(
                field=cond_data["field"],
                operator=cond_data["operator"],
                value=cond_data["value"],
            )

        return filter_obj


def build_source_filter(sources: list[str]) -> MetadataFilter:
    """Build a filter for specific sources."""
    return MetadataFilter().add_in("source", sources)


def build_date_filter(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    date_field: str = "created_at",
) -> MetadataFilter:
    """Build a date range filter."""
    return MetadataFilter().add_date_range(date_field, start_date, end_date)


def build_file_type_filter(file_types: list[str]) -> MetadataFilter:
    """Build a filter for specific file types."""
    return MetadataFilter().add_in("file_type", file_types)
