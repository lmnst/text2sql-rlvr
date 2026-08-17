"""Dataset preparation and prompt construction."""

from text2sql_rlvr.data.bird import (
    SPLITS,
    BirdExample,
    BirdSplit,
    discover_split,
    load_examples,
)
from text2sql_rlvr.data.prompt import (
    INSTRUCTION,
    INSTRUCTIONS,
    SYSTEM_PROMPT,
    PromptConfig,
    build_messages,
    build_user_prompt,
)
from text2sql_rlvr.data.schema import (
    Column,
    DatabaseSchema,
    ForeignKey,
    Table,
    fetch_sample_rows,
    format_schema,
    load_schema,
)
from text2sql_rlvr.data.schema_selection import (
    SCHEMA_MODES,
    SchemaSelection,
    foreign_key_graph,
    linked_table_names,
    oracle_table_names,
    render_selected_schema,
    required_tables_fk_connected,
    select_schema,
)

__all__ = [
    "INSTRUCTION",
    "INSTRUCTIONS",
    "SPLITS",
    "SYSTEM_PROMPT",
    "BirdExample",
    "BirdSplit",
    "Column",
    "DatabaseSchema",
    "ForeignKey",
    "PromptConfig",
    "SCHEMA_MODES",
    "SchemaSelection",
    "Table",
    "build_messages",
    "build_user_prompt",
    "discover_split",
    "fetch_sample_rows",
    "foreign_key_graph",
    "format_schema",
    "load_examples",
    "load_schema",
    "linked_table_names",
    "oracle_table_names",
    "required_tables_fk_connected",
    "render_selected_schema",
    "select_schema",
]
