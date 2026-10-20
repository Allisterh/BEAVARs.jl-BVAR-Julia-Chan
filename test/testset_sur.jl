T = 228;
n = 3;
p = 4;
k = n*p + 1
X = rand(T,k);
Xsur = BEAVARs.XSurFormMatrix(X,n);
Xsur_orig, Xsur_CI, X_CI = BEAVARs.SUR_form_dense(X,n);
Xtemp = rand(n,n);
Σt_inv = inv(Xtemp'*Xtemp);
Σ_inv_sp, Σt_LI              = BEAVARs.makeBlkDiag(T*n,n,0,Σt_inv);      # I(T) ⊗ Σ-1 and its indices for update

h                           = zeros(T,);                                # stochastic volatilities
expmh                       = exp.(-h);
Σ_inv_bb                    = BlockBandedMatrix{Float64}(undef,ones(Int,T,).*n, ones(Int,T,).*n, (0, 0))


XtΣ_inv_den                 = zeros(k*n,T*n);                           # will be X' ( I(T) ⊗ Σ-1 )   from page 6 in Chan 2020 LBA
XtΣ_inv                     = zeros(k*n,T*n);                           # will be X' ( I(T) ⊗ Σ-1 )   from page 6 in Chan 2020 LBA
XtΣ_inv_Xden                = zeros(n*k,n*k);                           # will be X' ( I(T) ⊗ Σ-1 ) X from page 6 in Chan 2020 LBA   
XtΣ_inv_X                   = zeros(n*k,n*k);                           # will be X' ( I(T) ⊗ Σ-1 ) X from page 6 in Chan 2020 LBA   


mul!(XtΣ_inv_den,Xsur_orig',Σ_inv_sp);                #  X'*( I(T) ⊗ Σ^{-1} )
mul!(XtΣ_inv,Xsur',Σ_inv_sp);                         #  X'*( I(T) ⊗ Σ^{-1} )

mul!(XtΣ_inv_X,XtΣ_inv_den,Xsur);
mul!(XtΣ_inv_Xden,XtΣ_inv_den,Xsur_orig);


@test Xsur == Xsur_orig                 # test whether the Xsur is defined correctly
@test XtΣ_inv_X ≈ XtΣ_inv_Xden          # test for whether the mul! is implemented correctly
@test XtΣ_inv_den == XtΣ_inv            # test for whether the mul! is implemented correctly for adjoint

xvec = rand(k*n)
@test isapprox(Xsur * xvec, Xsur_orig * xvec; atol=1e-12)   # test whether multiplication is correct

yvec = rand(T*n)
yvec_expected = 0.7 .* (Xsur_orig * xvec) .+ 0.2 .* yvec
@test isapprox(mul!(yvec, Xsur, xvec, 0.7, 0.2), yvec_expected; atol=1e-12) # test whether mul! multiplication is correct

@testset "adjoint SUR times block-banded matrix" begin
    Bdiag = BlockBandedMatrix{Float64}(undef, fill(n, T), fill(n, T), (0, 0))
    for t in 1:T
        Bdiag[Block(t, t)] .= Σt_inv
    end
    product = Xsur_orig' * Matrix(Bdiag)
    result = similar(product)
    @test mul!(result, Xsur', Bdiag) ≈ product

    initial = rand(size(result))
    result .= initial
    @test mul!(result, Xsur', Bdiag, 0.7, 0.2) ≈ 0.7 .* product .+ 0.2 .* initial

    Bband = BlockBandedMatrix{Float64}(undef, fill(n, T), fill(n, T), (1, 0))
    for s in 1:T, t in s:min(T, s + 1)
        Bband[Block(t, s)] .= rand(n, n)
    end
    @test mul!(similar(product), Xsur', Bband) ≈ Xsur_orig' * Matrix(Bband)
end

# Tests to the block-banded representation


BEAVARs.update_Σf_inv_bb!(Σ_inv_bb,Σt_inv,expmh,T,n);
# test whether the custom block-banded interface is the same as the sparse representation (exp(-h) = 1)
@test Matrix(Σ_inv_sp) ≈ Σ_inv_bb   


# test whether the multiplication of each block by h[element] is the same as the sparse representation (exp(-h) = 1)
h                           = zeros(T,).+1.01;  
expmh                       = exp.(-h);
BEAVARs.update_Σf_inv_bb!(Σ_inv_bb,Σt_inv,expmh,T,n);
for ij = 1:T
    copyto!(Σ_inv_sp.nzval,(ij-1)*n*n + 1: n*n + (ij-1)*n*n, Σt_inv.*expmh[ij],1:n*n)
end
@test Matrix(Σ_inv_sp) ≈ Σ_inv_bb   # test whether the custom block-banded interface is the same as the sparse representation when we mult(exp(-h) = 1)
