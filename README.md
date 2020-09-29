# imSitu_GAT
Graph Attention Network for Situation Recognition


# branch
## master (baseline_crf)
    \phi_i = \phi(i): image feature

    \psi_a(v; \phi_i): (504,)
    \psi_b(v, r, n; \phi_i): (504, 190, 10180)
    
    \psi_b(v, r; \phi_i) = \log \sum_n \exp \phi_b(v, r, n; \phi_i): (504, 190)

    \psi_b(v; \phi_i) = \log \sum_r \exp \phi_b(v, r; \phi_i): (504,)
    \phi(v; \phi_i) = \psi_a(v; \phi_i) + \psi_b(v; \phi_i): (504,)

    Z = \sum_v \phi(v; \phi_i): ()