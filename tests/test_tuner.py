from aibox.config import config_from_toml_stream


def test_search_space_resolution():
    from aibox.torch.tuning import gather_search_spaces_ray

    config_toml = """
    [tuner.search.model]
    masked.type = 'choice'
    masked.values = ['aligned', 'unaligned']
    
    mode.type = 'choice'
    mode.values = ['gray', 'rgb', 'label']

    [tuner.search.data.train]
    mode.type = 'choice'
    mode.values = ['aligned', 'unaligned']
    
    content_mode.type = 'choice'
    content_mode.values = ['gray', 'rgb', 'label']
    """

    config = config_from_toml_stream(config_toml)
    search_space = gather_search_spaces_ray(config)
    assert set(search_space.keys()) == {"model.masked", "model.mode", "data.train.mode", "data.train.content_mode"}
