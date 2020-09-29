# imSitu_GAT
Graph Attention Network for Situation Recognition


# branch
## master (baseline_crf)
    \phi_i = \phi(i): image feature

    \psi_a(v; \phi_i): (504,)
    \psi_b(v, Rv, RvN; \phi_i): (504, 190, 10180)

    <!-- p(v, r) = 0 for some r, v -->
    p_b(v, r; \phi_i) = \sum_n p_b(v, r, n;\phi_i)
    \psi_b(v, r; \phi_i) = \log \sum_n \exp \phi_b(v, r, n; \phi_i): (504, 190)
    <!-- p(n,v,r;\phi_i) = 0 for some v, r, n -->
    <!-- \psi(n,v,r;\phi_i) = {-\inf, 0} for some v, r, n -->

    p_b(v; \phi_i) = p_b(v, Rv;\phi_i) = \mul_{r\inRv} p_b(v, r;\phi_i)
    \psi_b(v; \phi_i) = \sum_{r\inRv} \phi_b(v, r; \phi_i): (504,)

    p(v; \phi_i) = p_a(v;\phi_i)*p_b(v;\phi_i)
    \phi(v; \phi_i) = \psi_a(v; \phi_i) + \psi_b(v; \phi_i): (504,)

    Z = \sum_v p(v; \phi_i) 
    log Z = \log \sum_v \exp \phi(v; \phi_i): ()

## role_crf
    \phi_i = \phi(i): image feature

    \psi_a(v; \phi_i): (504,)
    \psi_b(r, n; \phi_i): (190, 10180)
    
    \psi_b(r; \phi_i) = \log \sum_n \exp \phi_b(r, n; \phi_i): (190,)
                      = \sum_n p_b(v, r, n;\phi_i)
    \psi_b(v, r; \phi_i) = \sum_{r\inRv} \psi_b(r; \phi_i): (504, 6)
                      = \mul_{r\inRv} p_b(r;\phi_i)
    \psi_b(v; \phi_i) = \sum_r \phi_b(v, r; \phi_i): (504,)
                      = \mul_r p_b(v, r;\phi_i)

    \phi(v; \phi_i) = \psi_a(v; \phi_i) + \psi_b(v; \phi_i): (504,)
                    = p_a(v;\phi_i)*p_b(v;\phi_i)

    Z = \sum_v \phi(v; \phi_i): ()
    
## role_bottomup
    \phi_i = \phi(i): image feature

    \psi_a(r, v; \phi_i): (190, 504, )
    \psi_b(r, n; \phi_i): (190, 10180, )
    
    <!-- p(v, r; \phi_i) = 0 for some v, r -->
    p_a(v; \phi_i)  = \sum_r p_a(v, r; \phi_i)
                    = \sum_{\r\in\Rv} p_a(v, r; \phi_i)
    <!-- \phi(v, r; \phi_i) = -\inf for some v, r -->
    \psi_a(v; \phi_i) = \log \sum_r \exp \phi_a(v, r; \phi_i): (504,)
                      = \log \sum_{\r\inRv} \exp \phi_a(v, r; \phi_i): (504,)

    \psi_b(r; \phi_i) = \log \sum_n \exp \phi_b(r, n; \phi_i): (190,)
                      = \sum_n p_b(v, r, n;\phi_i)
    \psi_b(v, r; \phi_i) = \sum_{r\inRv} \psi_b(r; \phi_i): (504, 6)
                      = \mul_{r\inRv} p_b(r;\phi_i)
    \psi_b(v; \phi_i) = \sum_r \phi_b(v, r; \phi_i): (504,)
                      = \mul_r p_b(v, r;\phi_i)

    \phi(v; \phi_i) = \psi_a(v; \phi_i) + \psi_b(v; \phi_i): (504,)
                    = p_a(v;\phi_i)*p_b(v;\phi_i)

    Z = \sum_v \phi(v; \phi_i): ()
