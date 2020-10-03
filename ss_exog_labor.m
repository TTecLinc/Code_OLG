% Made by Peilin Yang 12/01/2019
% For my paper: 
% China’s Policy Instruments : Tax Reduction, Retirement Prolonging and Welfare Changes


function [ y ] = ss_exog_labor(x)
% computes the foc's for the case with exogenous labor 
% and number of periods equal to nw+iage

    def_global_USdebt
	y=x;
	asset=[0;x;0];		% assets of the cohorts
	k=kshare*asset;
	b=(1-kshare)*asset;
	labors=lbar*ones(nw,1);
	if iage>0
		labors=[labors; zeros(iage,1)];
    end
	
	
	% consumption
	c=zeros(nage,1);	
	
	for i=1:1:nw
		c(i) = (1-taun-taup)*wbar*ef(i)*labors(i)+(1-tauk)*(dbar-delta)*k(i)+k(i)-ygrowth*k(i+1)+rbbar*b(i)-ygrowth*b(i+1)+trbar;
		c(i) = c(i)/(1+tauc);
    end

	if iage>0
		for i=1:1:iage-1
			c(i+nw) = pen+(1-tauk)*(dbar-delta)*k(nw+i)+k(nw+i)-ygrowth*k(nw+i+1)+rbbar*b(nw+i)-ygrowth*b(nw+i+1)+trbar;
			c(i+nw) = c(i+nw)/(1+tauc);
        end
		c(iage+nw) = pen+(1-tauk)*(dbar-delta)*k(nw+iage)+k(nw+iage)+rbbar*b(nw+iage)+trbar;
		c(iage+nw) = c(nw+iage)/(1+tauc);
    end
	
	% intertemporal first-order conditions
	for i=1:1:nage-1
		y(i)=ygrowth^(eta1)*uc(c(i),labors(i))/uc(c(i+1),labors(i+1))-sp(i)*beta1*(1+(1-tauk)*(dbar-delta));
    end
	
end

