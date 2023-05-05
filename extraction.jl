using LinearAlgebra
using Random
using Plots
using Statistics


global Noise_lvl = [0.00,1.00]
global Noise_lvcont = LinRange(0.00,1.00,20)
global ev = 1.6*10^(-19)
global hbar = 1.057*10^(-34)
global kb = 1.38*10^(-23)

#Extracting the Urbach Energy from DOSCAR (obtained via VASP)


function extract_DOS(filepath)
    nheader = 8 # number of "header" lines in the DOSCAR files
    nline = 0
    DOS = []
    Eng = []
    DOSfile = open(filepath, "r")
    while !eof(DOSfile)
        if nline < nheader
            readline(DOSfile)
        else
            line = readline(DOSfile)
            DOSxEng = split(line)
            DOSval = parse(Float64, DOSxEng[2])
            Engval = parse(Float64, DOSxEng[1])
            push!(DOS, DOSval)
            push!(Eng, Engval)
        end
        nline += 1
    end
    close(DOSfile)
    return DOS, Eng
end


function urbachenergy(sigma1,DOS1,sigma2,DOS2) 
    mobility_edge = 0
    lowest_eng = 0
    j = 1
    for i in 1:length(sigma1) 
        if DOS1[i]!=0
            mobility_edge = sigma1[i]
            break
        end 
    end 
    while DOS2[j]==0 
        j+=1
    end 
    lowest_eng = sigma2[j]
    return mobility_edge - lowest_eng #Urbach energy (mobility edge - energy of the least energetic localised state)
end 

#DOS1Material,eng1Material  = extract_DOS("C:\\Users\\DOSCAR PATH")
#DOS2Material,eng2Material = extract_DOS("C:\\Users\\DOSCAR2 PATH")
#urbachenergy(eng1Material,DOS1Material,eng2Material,DOS2Material)
