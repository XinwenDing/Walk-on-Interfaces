function [V, F, domain_layout, root_id] = load_domain_info(domain_name)
    [V{1}, F{1}] = readOBJ("../../../OBJ2D/" + domain_name + "/circle1.obj");
    [V{2}, F{2}] = readOBJ("../../../OBJ2D/" + domain_name + "/circle2.obj");
    [V{3}, F{3}] = readOBJ("../../../OBJ2D/" + domain_name + "/curvedstar1.obj");

    [domain_layout, root_id] = configure_domain_layout();
end