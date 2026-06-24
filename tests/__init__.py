# Tests package marker.
#
# Required so tests can do `from tests._mock_embedder import MockEmbeddings`.
# Without this, `tests` is just a namespace package and the import falls
# back to a relative lookup that doesn't find sibling files.
