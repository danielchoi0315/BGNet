from bgnet.geometry import DEFAULT_SOURCE_NAMES, build_default_source_layout, canonicalize_channel_name, normalized_source_adjacency


def test_geometry_constants_match_expected_shape():
    layout = build_default_source_layout()
    assert layout.names == DEFAULT_SOURCE_NAMES
    assert layout.positions.shape == (32, 3)
    adjacency = normalized_source_adjacency(layout.positions)
    assert adjacency.shape == (32, 32)


def test_channel_canonicalization_matches_known_aliases():
    assert canonicalize_channel_name("EEG FP1-REF") == "Fp1"
    assert canonicalize_channel_name("T3") == "T7"
    assert canonicalize_channel_name("CZ") == "Cz"

