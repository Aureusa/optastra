from optastra.backbones._registry import get_backbone_entrypoint, list_backbones, register_backbone, get_backbone_module

def test_backbone_registry():
    # Define a dummy backbone function for testing
    @register_backbone
    def DummyBackbone():
        pass

    # Test if the backbone is registered
    assert "DummyBackbone" in list_backbones(), "DummyBackbone should be registered."

    # Test if the entrypoint can be retrieved
    entrypoint = get_backbone_entrypoint("DummyBackbone")
    assert entrypoint == DummyBackbone, "Entrypoint should match the registered function."

    # Test if the module name can be retrieved
    module_name = get_backbone_module("DummyBackbone")
    assert module_name == __name__.split('.')[-1], "Module name should match the current module."

def test_backbone_registry_duplicate_registration():
    # Define a dummy backbone function for testing
    @register_backbone
    def AnotherDummyBackbone():
        pass

    # Attempt to register the same backbone again and expect a ValueError
    try:
        @register_backbone
        def AnotherDummyBackbone():
            pass
        assert False, "Duplicate registration should raise ValueError."
    except ValueError as e:
        assert str(e) == f'backbone AnotherDummyBackbone already registered by {__name__.split(".")[-1]}', "Error message should indicate duplicate registration."

def test_backbone_registry_nonexistent_entrypoint():
    # Attempt to retrieve an entrypoint for a non-existent backbone and expect a ValueError
    try:
        get_backbone_entrypoint("NonExistentBackbone")
        assert False, "Retrieving a non-existent entrypoint should raise ValueError."
    except ValueError as e:
        assert str(e) == 'backbone NonExistentBackbone is not registered', "Error message should indicate non-existent backbone."

def test_backbone_registry_nonexistent_module():
    # Attempt to retrieve a module for a non-existent backbone and expect a ValueError
    try:
        get_backbone_module("NonExistentBackbone")
        assert False, "Retrieving a non-existent module should raise ValueError."
    except ValueError as e:
        assert str(e) == 'backbone NonExistentBackbone is not registered', "Error message should indicate non-existent backbone."

def test_backbone_registry_list_with_filter():
    # Define a dummy backbone function for testing
    @register_backbone
    def FilteredDummyBackbone():
        pass

    # Test listing backbones with a filter that matches the registered backbone
    filtered_list = list_backbones(filter="Filtered*")
    assert "FilteredDummyBackbone" in filtered_list, "FilteredDummyBackbone should be in the filtered list."

    # Test listing backbones with a filter that does not match any registered backbone
    filtered_list_empty = list_backbones(filter="NonMatching*")
    assert len(filtered_list_empty) == 0, "Filtered list should be empty for non-matching filter."

def test_backbone_registry_list_with_module():
    # Define a dummy backbone function for testing
    @register_backbone
    def ModuleDummyBackbone():
        pass

    # Test listing backbones with the current module name
    module_name = __name__.split('.')[-1]
    module_list = list_backbones(module=module_name)
    assert "ModuleDummyBackbone" in module_list, "ModuleDummyBackbone should be in the list for the current module."

    # Test listing backbones with a non-existent module name
    non_existent_module_list = list_backbones(module="NonExistentModule")
    assert len(non_existent_module_list) == 0, "List should be empty for a non-existent module."

def test_backbone_registry_list_with_module_and_filter():
    # Define a dummy backbone function for testing
    @register_backbone
    def ModuleFilteredDummyBackbone():
        pass

    # Test listing backbones with the current module name and a matching filter
    module_name = __name__.split('.')[-1]
    filtered_module_list = list_backbones(module=module_name, filter="ModuleFiltered*")
    assert "ModuleFilteredDummyBackbone" in filtered_module_list, "ModuleFilteredDummyBackbone should be in the filtered list for the current module."

    # Test listing backbones with the current module name and a non-matching filter
    non_matching_filtered_module_list = list_backbones(module=module_name, filter="NonMatching*")
    assert len(non_matching_filtered_module_list) == 0, "Filtered list should be empty for a non-matching filter in the current module."

def test_backbone_registry_list_with_nonexistent_module_and_filter():
    # Test listing backbones with a non-existent module name and a filter
    non_existent_module_list = list_backbones(module="NonExistentModule", filter="*")
    assert len(non_existent_module_list) == 0, "List should be empty for a non-existent module even with a filter."

def test_backbone_registry_list_with_nonexistent_module_and_non_matching_filter():
    # Test listing backbones with a non-existent module name and a non-matching filter
    non_existent_module_list = list_backbones(module="NonExistentModule", filter="NonMatching*")
    assert len(non_existent_module_list) == 0, "List should be empty for a non-existent module and a non-matching filter."

def test_backbone_registry_list_with_empty_filter():
    # Define a dummy backbone function for testing
    @register_backbone
    def EmptyFilterDummyBackbone():
        pass

    # Test listing backbones with an empty filter (should return all registered backbones)
    all_backbones = list_backbones(filter="")
    assert "EmptyFilterDummyBackbone" in all_backbones, "EmptyFilterDummyBackbone should be in the list when using an empty filter."

def test_backbone_registry_list_with_none_filter():
    # Define a dummy backbone function for testing
    @register_backbone
    def NoneFilterDummyBackbone():
        pass

    # Test listing backbones with a None filter (should return all registered backbones)
    all_backbones = list_backbones(filter=None)
    assert "NoneFilterDummyBackbone" in all_backbones, "NoneFilterDummyBackbone should be in the list when using a None filter."

def test_backbone_registry_list_with_none_module():
    # Define a dummy backbone function for testing
    @register_backbone
    def NoneModuleDummyBackbone():
        pass

    # Test listing backbones with a None module (should return all registered backbones)
    all_backbones = list_backbones(module=None)
    assert "NoneModuleDummyBackbone" in all_backbones, "NoneModuleDummyBackbone should be in the list when using a None module."

def main():
    # Run all the test functions
    test_backbone_registry()
    test_backbone_registry_duplicate_registration()
    test_backbone_registry_nonexistent_entrypoint()
    test_backbone_registry_nonexistent_module()
    test_backbone_registry_list_with_filter()
    test_backbone_registry_list_with_module()
    test_backbone_registry_list_with_module_and_filter()
    test_backbone_registry_list_with_nonexistent_module_and_filter()
    test_backbone_registry_list_with_nonexistent_module_and_non_matching_filter()
    test_backbone_registry_list_with_empty_filter()
    test_backbone_registry_list_with_none_filter()
    test_backbone_registry_list_with_none_module()

if __name__ == "__main__":
    main()
