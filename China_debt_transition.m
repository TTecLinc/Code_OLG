% Made by Peilin Yang 12/01/2019
% For my paper: 
% China’s Policy Instruments : Tax Reduction, Retirement Prolonging and Welfare Changes

clear all;
close all;
clc;
tic;
%def_global
%def_global_transition
%fhandle_function1 = @rftr;

run = 1;       
if run==1
    survivalprobs=xlsread('survival_probs_China.xlsx','A23:P42');
    popgrowth=xlsread('survival_probs_China.xlsx','A47:P47');
    timespan=linspace(2020,2100,16);
    nrate=[timespan' popgrowth'];
    save('survivalprobs','popgrowth','nrate');
else
    load('survivalprobs','popgrowth','nrate');
end

%% Step 1 : Set Para

year1=1995;				
yearinitial=year1;
%UN set
movavperiods=4;			
case_tau=0;             % 1 -- labor income tax, 2 -- capital income tax, 3 -- consumption tax adjusts to balance budget
                        % 0 -- extra taxes are transfered lump-sum
case_tauc=1;            % 0 -- consumption tax=10%, 1 -- consumption tax=5%
case_repl=1;            % 1 -- replacement ratio = benchmark, 0 -- reduction by 10 percentage points
case_retirement=1;      % 1 -- retirement at age 65, 0 -- retirement at age 70

case_productivity=0;	% 1 -- all agents have productivity equal to one, 0 -- hump-shaped age-productivity profile
case_growth=1;			% 1 -- growth, 0 -- no growth
periodlength=5;			% 5 -- 5 years, 1 -- 1 year
%periodlength=1;
case_pen=1;				% 1 -- calibration such that replacement ratio is equal to empirical one			
						% 0 -- pension contribution rate equal to empirical one
case_level_pen=0;		% 1 -- level of pension remains at 2010 level, 0 -- repl ratio remains constant
case_level_pen1=0;
                        
case_UNscen=1;          % 1 -- medium variant, 2 -- low variant, 3 -- high variant for population projection UN (2015)
case_beta_calib=1;		% 1 -- benchmark case, the calibration of beta

benchmark=1;			% 1 -- benchmark case: saves the amount of transfers 

save_results=1;         % 1 -- save results
maxit=50;               % maximum number of iterations over aggregate capital stock
phi=0.9;                % updating parameter
tol = 1e-10;            % tolerance

% computational parameters transitionrfinalyear=2095;	
finalyear=2050;	% must be in {2050,2055,..,2095} 
				% 0 -- benchmark, transfers are constant and labor income tax adjusts
policy=2;		% 1 -- transfers adjust, 
				% 2 -- during the first ndebt periods, debt adjusts. Afterwards, labor income tax rate
				% 3 -- during the first ndebt periods, debt adjusts. Afterwards, transfers
				
nt=30;			% number of transition periods starting in t=2010,..
update=1;		% 1 -- linear update
ndebt=8;		% number of periods during the transition where extra expenditures are financed by debt
%ndebt=10;		% does not converge for policy=2
tolt= 1e-5;     % tolerance for transition

if periodlength==1
	nage=75;
	Rage=46;            % first period of retirement 
	nr=nage-Rage+1;     % number of retirement years 
	nw=Rage-1;          % number of working years 	
elseif periodlength==5
	nage=15;
	Rage=10;            % first period of retirement 
	nr=nage-Rage+1;     % number of retirement years 
	nw=Rage-1;          % number of working years 	
else
	disp('wrong parameter period length');
end

%% Step2 : Efficiency Profile

nage1=75;
age=linspace(20,94,nage1);
if run==1
    efage=xlsread('efficiency_profile.xlsx',1,'A1:A45');
    efage=efage/mean(efage);
    save('efage');
else
    load('efage');
end
    
year0=(year1-1950)/5+1;

if year0==round(year0)
	popgrowth=nrate(year0-movavperiods+1:year0,case_UNscen+1);
	popgrowth=mean(popgrowth);
	popgrowth=popgrowth/100;        
	sp=survivalprobs(1:nage,year0-movavperiods+1:year0);
	sp=mean(sp,2);
else
	disp('year1 must be a multiple of 5');
end

sp0 = sp;

if periodlength==5;		
	if case_productivity==0
		efage1=zeros(nw,1);
		for i=1:1:nw
			efage1(i)=mean(efage((i-1)*5+1:i*5));		% average productivity
        end
		efage=efage1;
	elseif case_productivity==1
		efage=ones(nw,1);
    else
		disp('wrong parameter case_productivity');
    end
end

ef=efage;
%% Step 3 : Calibration
alpha1=0.35;
delta=0.083;
rbbar=1.04;		% annual real interest rate on bonds
taun=0.28;
taunbar=taun;
taulbar=0.28;		% both taul+taup=0.28!!, see Mendoza, Razin (1994), p. 311
tauk=0.36;
taukbar=tauk;
if case_tauc==1
    tauc=0.05;
else
    tauc=0.10;
end
taucbar=tauc;
taup=0.124;
taupbar=0.124;		
if case_repl==1
    replacement_ratio=0.352;    % gross replacement ratio
else 
    replacement_ratio=0.352-0.1;
end
bybar=0.63;
%bybar=1.049;
%bybar=1;
if year1==2010
	gybar=0.18;
else
	gybar=0.239;
	gybar=0.18;
end

% Frisch labor supply elasticity
varphi=0.3;
lbar=0.25;		% steady-state labor supply
eta1=2.0;		% 1/IES
kappa=21.5;

if case_growth==1
	ygrowth=1.02;		% annual growth factor
else
	ygrowth=1.00;
end

if periodlength==5      % transformation of annual paramaters to 5-year values
	delta=1-(1-delta)^5;
	rbbar=rbbar^5;
	ygrowth=ygrowth^5;
	bybar=bybar/5;
	popgrowth=(1+popgrowth)^5-1;
end

debtoutputratio=bybar;
% load('Ch7_US_debt','beta1');
beta1 = 1.2740392;

% computation of cohort mass
mass=ones(nage,1);
for i=2:1:nage
	mass(i)=mass(i-1)*sp(i-1)/(1+popgrowth);
end
mass=mass/sum(mass);
massinitial=mass;

% load benchmark values: first run Ch7_US_debt.m
load('trbench','ybench','cbench','lbench','penbench','taxesbench','xstartss','bbench','gbench');
trbar=trbench;
ybar=ybench;
c=cbench;
labors=lbench;
pen=penbench;
taxes=taxesbench;
x0=xstartss;
bigb=bbench;
bigg=gbench;

utilityss=utility(c,labors);
utilityss


x00=x0;
if nw==10       % add guess for labor supply of the nw-old worker 
                % for the case that retirement period is one period later 
		ntemp=size(x0,1);
		x00=[x0(1:nage+9); 0.3; x0(nage+10:ntemp)];
		x0=x00;
end
	
% initialisation
taun1=taunbar;
tauk1=taukbar;