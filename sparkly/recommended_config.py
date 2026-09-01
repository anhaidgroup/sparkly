import json
from collections import defaultdict
from typing import Iterator

from sparkly.index_config import IndexConfig
from sparkly.query_generator.query_spec import QuerySpec


class RecommendedConfig:
    """
    User-facing Sparkly Auto configuration.

    A RecommendedConfig contains one or more recommended configurations and
    serializes to a JSON list.
    """

    def __init__(self, *, configs: list[dict]):
        if not isinstance(configs, list):
            raise TypeError("configs must be a list")

        if not configs:
            raise ValueError("Recommended config list cannot be empty")

        if not all(isinstance(config, dict) for config in configs):
            raise TypeError("Each recommended config must be a dictionary")

        self.configs = configs

    @classmethod
    def from_components(
        cls,
        *,
        index_config: IndexConfig,
        query_specs: list[QuerySpec],
    ) -> "RecommendedConfig":
        if not query_specs:
            raise ValueError("query_specs cannot be empty")

        configs = []

        for query_spec in query_specs:
            query_spec_dict = query_spec.to_dict(json_safe=True)

            configs.append({
                "id_col": index_config.id_col,
                "concat_fields": index_config.concat_fields,
                "spec": query_spec_dict["spec"],
                "boost_map": query_spec_dict.get("boost_map", {}),
            })

        return cls(configs=configs)

    @classmethod
    def from_dicts(cls, configs: list[dict]) -> "RecommendedConfig":
        return cls(configs=configs)

    @classmethod
    def from_json(cls, data: str | list[dict]) -> "RecommendedConfig":
        if isinstance(data, str):
            data = json.loads(data)

        if not isinstance(data, list):
            raise TypeError(
                "Recommended config JSON must contain a list"
            )

        return cls.from_dicts(data)

    def to_dicts(self) -> list[dict]:
        return self.configs

    def to_json(self) -> str:
        return json.dumps(self.to_dicts())

    def to_components(self) -> list[tuple[IndexConfig, QuerySpec]]:
        return [
            self._config_to_components(config)
            for config in self.configs
        ]

    def to_index_configs(self) -> list[IndexConfig]:
        return [
            self._config_to_index_config(config)
            for config in self.configs
        ]

    def to_query_specs(self) -> list[QuerySpec]:
        return [
            self._config_to_query_spec(config)
            for config in self.configs
        ]

    def __len__(self) -> int:
        return len(self.configs)

    def __iter__(self) -> Iterator[dict]:
        return iter(self.configs)

    @classmethod
    def _config_to_components(
        cls,
        config: dict,
    ) -> tuple[IndexConfig, QuerySpec]:
        return (
            cls._config_to_index_config(config),
            cls._config_to_query_spec(config),
        )

    @classmethod
    def _config_to_index_config(
        cls,
        config: dict,
    ) -> IndexConfig:
        index_config = IndexConfig()
        index_config.id_col = config["id_col"]

        for field, source_fields in config.get(
            "concat_fields",
            {},
        ).items():
            index_config.concat_fields[field] = source_fields

        field_to_analyzers = cls._infer_field_to_analyzers(
            config["spec"]
        )

        for field, analyzers in field_to_analyzers.items():
            index_config.add_field(field, analyzers)

        return index_config

    @staticmethod
    def _config_to_query_spec(config: dict) -> QuerySpec:
        return QuerySpec.from_dict({
            "spec": config["spec"],
            "boost_map": config.get("boost_map", {}),
            "filter": [],
        })

    @staticmethod
    def _infer_field_to_analyzers(spec: dict) -> dict:
        field_to_analyzers = defaultdict(set)

        for query_fields in spec.values():
            for query_field in query_fields:
                field, analyzer = query_field.rsplit(".", 1)
                field_to_analyzers[field].add(analyzer)

        return {
            field: sorted(analyzers)
            for field, analyzers in field_to_analyzers.items()
        }