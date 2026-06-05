function [V, F, domain_layout, root_id] = load_domain_info(domain_name)
    [V{1}, F{1}] = readOBJ("../../../OBJ3D/" + domain_name + "/sphere.obj");
    [V{2}, F{2}] = readOBJ("../../../OBJ3D/" + domain_name + "/spot.obj");
    
    [domain_layout, root_id] = configure_domain_layout();
end