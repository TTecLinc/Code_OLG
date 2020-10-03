function [ y ] = interest( K,L )
% computes the interest rate

    def_global_USdebt
    y = alpha1* (K.^(alpha1-1)).* (L.^(1-alpha1));


end

