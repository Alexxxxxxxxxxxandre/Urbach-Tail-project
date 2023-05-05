using DSP
using Distributions
using IterativeSolvers
using LinearAlgebra
using Noise
using Random
using Plots


global e = 1.6*10^-19
global eps0 = 4*π*10^-7
global m = 9.11e-31 * 0.98 #in silicon
global hbar = 6.63e-34/(2*π)
global npts = 1000
global Noise_lvl = [0.01,0.1,0.2,0.5,1.00]
global ev = 1.6*10^(-19)
global nEng = 50
#Inspired by Dr Oulton's Assessed Problem Sheet 2 (tight binding model simulation) 


function wells!(V0,W,Th,V,d,N) 
    for i in 1:2*N 
        if i%2 == 1
            V[i] = 0
            d[i] = W*10^(-9)  
        else 
            V[i] = V0*ev
            d[i] = Th*10^(-9) 
        end
    end  
    return V,d       
end


function dist(d) 
    sumd = zeros(Float64,length(d)) 
    sumd[1] = d[1]
    for k in 2:length(d)  
        sumd[k] = sumd[k-1] + d[k]
    end 
return sumd 
end 

function TransfMatrix2lay(E, V1, V2)
    if E >= V1
        k1 = sqrt(E-V1)
    elseif  E < V1
        k1 = 1im*sqrt(V1-E)
    end
    if E >= V2
        k2 = sqrt(E-V2)
    elseif E < V2
        k2 = 1im*sqrt(V2-E)
    end
        
    m = k1/k2
    T2 = 0.5 * Array([1+m 1-m; 1-m 1+m])
    return T2 
end

#Transfer matrix within a layer at the same potential energy
function TransfMatrix1layer(E, V, d)
    k = Array([1])
    if E >= V
        k = d*1im*sqrt(2*e*m)/hbar * sqrt(E-V)
    elseif E < V
        k = -d*sqrt(2*e*m)/hbar * sqrt(V-E)
    end
        
    T1 = Array([exp(k) 0 ; 0 exp(-k)])
    return T1 
end


function Wavefunct(E,d,V)
    T = Matrix(1.0I,2,2) 
    wvfct = zeros(Complex{Float64},npts-1)
    #In this loop, we obtain the transmission and reflection coefficients 
    for j in 1:length(V)
        if j == length(V)
            T2 = TransfMatrix2lay(E, V[j], V[1])  #periodic boundary 
            T = T2 * T
            T1 = TransfMatrix1layer(E, V[j], d[j])
            T = T1 * T
    
        else
            T2 = TransfMatrix2lay(E, V[j], V[j+1])  
            T = T2 * T
            T1 = TransfMatrix1layer(E, V[j+1], d[j+1])
            T = T1 * T
        end
    end    
    r = -T[2,1]/T[2,2]
    t = T[1,1]+T[1,2]*r
    
    z = LinRange(0, sum(d), npts) # position of the electron  /1st well
    T = Matrix(1.0I,2,2) 
    Layinit = 1
    distlay = dist(d)
    dz = z[2]-z[1]
    v = Array([1, r])
    for k in 1:(length(z)-1)
        if z[k+1] > distlay[Layinit] 
            if Layinit > length(V) - 1 
                p = Layinit - length(V) + 1
                T1 = TransfMatrix1layer(E, V[Layinit], distlay[Layinit]-z[k])
                T = T1 * T
                T2 = TransfMatrix2lay(E, V[Layinit], V[p])
                T = T2 * T
                T1 = TransfMatrix1layer(E, V[p], z[k+1]-distlay[Layinit])
                T = T1 * T 
                Layinit += 1
            
            else 
                T1 = TransfMatrix1layer(E, V[Layinit], distlay[Layinit] - z[k])
                T = T1 * T
                T2 = TransfMatrix2lay(E, V[Layinit], V[Layinit + 1])
                T = T2 * T
                T1 = TransfMatrix1layer(E, V[Layinit+1], z[k+1] - distlay[Layinit])
                T = T1 * T 
                Layinit += 1
            end
        
        elseif z[k+1] < distlay[Layinit] 
            T1 = TransfMatrix1layer(E, V[Layinit], dz)
            T = T1 * T
            
        elseif z[k+1] == distlay[Layinit] 
            if Layinit < length(V) 
                T1 = TransfMatrix1layer(E, V[Layinit], dz)
                T = T1 * T 
                T2 = TransfMatrix2lay(E, V[Layinit], V[Layinit+1])
                T = T2 * T
                Layinit += 1    
            end
        end
        a1 = T[1,1]*v[1] + T[1,2]*v[2]
        a2 = T[2,1]*v[1] + T[2,2]*v[2]
        wvfct[k] = a1 + a2
    end
    return wvfct 
end

function wvfct(E,d,V,i) #wavefunction of the electron in the ith well
    if i<length(V)
        VtruncR = V[2*i - 1:length(V)]
        VtruncL = reverse(V[1:2*i - 1])
        dtruncL = reverse(d[1:2*i - 1])
        dtruncR = d[2*i - 1:length(d)]
        wvRtrunc = Wavefunct(E,dtruncR,VtruncR)
        wvLtrunc = Wavefunct(E,dtruncL,VtruncL)
    else 
        j = i - length(V)
        VtruncR = V[2*j - 1:length(V)]
        VtruncL = reverse(V[1:2*j - 1])
        dtruncL = reverse(d[1:2*j - 1])
        dtruncR = d[2*j - 1:length(d)]
        wvRtrunc = Wavefunct(E,dtruncR,VtruncR)
        wvLtrunc = Wavefunct(E,dtruncL,VtruncL)
    end
    wvfc = copy(cat(wvLtrunc,wvRtrunc,dims=1))
    return wvfc 
end 


function hopping(V,d,E,W,Th) #hopping between the ith well and i+1 th well
    z = LinRange(0, sum(d), npts)
    dz = z[2] - z[1]
    distlay = dist(d)
    hopp = zeros(Complex{Float64},trunc(Int64,length(V)/2))
    lWell = trunc(Int64,2*npts*W/(length(V)*(W+Th))) #relative length of the barrier (to the whole structure)
    lBarr = trunc(Int64,2*npts*Th/(length(V)*(W+Th))) #relative length of the barrier
    for i in 1:trunc(Int64,length(hopp))
        hoppi = 0 
        wvfL = wvfct(E,d,V,i) #wavefunction in the ith well
        if i==1    
            wvfR = wvfct(E,d,V,i+1) #wavefunction in the next well 
            integrd = zeros(Complex{Float64},length(wvfL))
            integrd = conj(wvfL) .* wvfR * V[2*i] 
            truncintg = integrd[1:2*lWell + lBarr] 
        elseif i==length(hopp)
            wvfR = wvfct(E,d,V,1) 
            integrd = zeros(Complex{Float64},length(wvfL))
            integrd = conj(wvfL) .* wvfR * V[2*i] 
            truncintg = cat(integrd[(2*length(hopp) - 1)*(lWell+lBarr):length(integrd)],integrd[1:lWell],dims=1)
        else 
            wvfR = wvfct(E,d,V,i+1) 
            integrd = zeros(Complex{Float64},length(wvfL))
            integrd = conj(wvfL) .* wvfR * V[2*i] 
            truncintg = integrd[(i-1)*(lWell + lBarr):(i+1)lWell + i*lBarr ] 
        end 
        
        for j in 1:length(truncintg)
            hoppi += truncintg[j]*dz
        end
        hopp[i] = hoppi
    end 
    return hopp 
end 


function whiteNoise(V,Noiselv) 
    WngV = zeros(Float64,length(V))
    for i in 1:length(V) 
        if i%2 == 0 
            WngV[i] = V[i] + Noiselv*rand()*maximum(V)/2
        end 
    end #Noiselv is between 0 and 1 
return WngV
end

function init_H(V,t) 
    Vtrunc = V[2:2:length(V)]
    tboundary = pop!(t)
    H = zeros(Complex{Float64},length(Vtrunc),length(Vtrunc)) 
    for i in 1:length(Vtrunc)
        for j in 1:length(Vtrunc) 
            if i==j 
                H[i,j] = Vtrunc[i] 
            elseif i==j-1 
                H[i,j] = t[i]
            elseif i==j+1
                H[i,j] = t[j] 
            end 
        end 
    end 
    H[length(Vtrunc),1],H[1,length(Vtrunc)] = tboundary,tboundary 
    return H 
end


function Gaussn(m,sig,x)  
    return exp(-(x-m)^2/(2*sig^2))/(sig*sqrt(2*π))
end


function getDOS(WngV,max,E) 
    sort!(E) 
    m = 0 
    sig = 1 #needs to be large enough so that 2 energy states that are close to each other have a "true" contribution to DOS
    l = trunc(Int64,400*max)
    yE  = zeros(Float64,l)
    for j in 1:length(E)
        for i in 1:l
        if trunc(Int64,100*E[j]/ev) == trunc(Int64,i - l/2) 
            yE[i] += 1 
        end
        end
    end
    contgauss = Normal(m,sig) 
    bot,top = quantile(contgauss,[0.01,0.99])
    diff = top - bot
    xGauss = LinRange(bot,top,l) 
    yGauss = zeros(Float64,l)
    for i in 1:l
        yGauss[i] = Gaussn(m,sig,xGauss[i]) 
    end
    DOS = conv(yE,yGauss) 
    return DOS , diff
end 


function Tail(V0,W,Th,N) 
    V = zeros(Float64,2*N)
    d = zeros(Float64,2*N)
    V,d = wells!(V0,W,Th,V,d,N)[1] , wells!(V0,W,Th,V,d,N)[2]
    max = maximum(V)/ev
    Eng = LinRange(0.01,2*maximum(V),nEng)
    E = zeros(Float64,0)
    H = Matrix(1.0I,length(V),length(V))
    for i in 1:length(Noise_lvl)
        print("\n i------",i)
        #t = zeros(Complex{Float64},trunc(Int64,length(V)/2))
        WngV = whiteNoise(V,Noise_lvl[i])
        Totenerg  = zeros(Float64,0)
        for p in 1:length(Eng)  #what do we do with the t for each trial energy?
            t = hopping(V,d,Eng[p],W,Th) 
            H = init_H(WngV,t) 
            E = real.((eigvals(H)))
            Totenerg = cat(Totenerg,E,dims=1)
        end 
        print("\n Totenerg : ",sort!(Totenerg/ev))
        DOS,diff= getDOS(WngV,max,Totenerg)[1],getDOS(WngV,max,Totenerg)[2]
        x = LinRange(minimum(E)/ev - diff/2,maximum(E) + diff/2,length(DOS))
        xlabel!("Energy") 
        ylabel!("Density of states") 
        if i==1 
            plot(x,DOS,title ="1D Density of states dependance on noise percentage", label = "0")
        else 
            p = plot!(x,DOS, label = trunc(Int64,100*Noise_lvl[i]), xlim=(1,2.5))
            display(p)
        end
    end
end
