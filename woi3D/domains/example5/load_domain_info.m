function [V, F, domain_layout, root_id] = load_domain_info(domain_name)
    [V{1}, F{1}] = readOBJ("../../../OBJ3D/" + domain_name + "/landscape.obj");
    [V{2}, F{2}] = readOBJ("../../../OBJ3D/" + domain_name + "/rock4_high_res.obj");
    [V{3}, F{3}] = readOBJ("../../../OBJ3D/" + domain_name + "/rock1_high_res.obj");
    [V{4}, F{4}] = readOBJ("../../../OBJ3D/" + domain_name + "/rock5_high_res.obj");
    [V{5}, F{5}] = readOBJ("../../../OBJ3D/" + domain_name + "/rock3_high_res.obj");
    [V{6}, F{6}] = readOBJ("../../../OBJ3D/" + domain_name + "/rock6_high_res.obj");
    [V{7}, F{7}] = readOBJ("../../../OBJ3D/" + domain_name + "/rock2_high_res.obj");
    
    [domain_layout, root_id] = configure_domain_layout();
end