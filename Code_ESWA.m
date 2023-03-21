%Step 1: Original data

%number of experts
q = 4;

%number of alternatives
n = 5;

%number of criteria
l = 5;

%indexes cost criteria
i_cost = [1 2 3];

%indexes benefit criteria
i_benf = [4 5];

%Granularity experts
M_g = [6 4 6 4];

%Initial preferences
Expert1 = {[4] [1] [5] [2] [5 6]; [4] [6] [4] [4 5] [3]; [4 5] [3 4] [4 5] [4] [2]; [2] [4] [2 3] [5] [6]; [1 2] [2] [4] [5] [6]};

Expert2 = {[3] [2] [2] [4] [2 3]; [2] [3] [1] [2] [4]; [2 3] [1] [1 2] [3] [3]; [2] [1 2] [3] [4] [2]; [1 2] [2] [2] [2] [3]};

Expert3 = {[5] [2] [3] [2] [3 4]; [4] [4] [2] [3 4] [2 3]; [4] [6] [4] [4] [4]; [2 3] [1] [5] [1] [5]; [5] [4] [1] [5] [4]};

Expert4 = {[3] [1] [1] [4] [3 4]; [4] [2] [2] [4] [2 3]; [2] [1] [1 2] [4] [3]; [3 4] [2] [2 3] [4] [2]; [1] [2] [1] [2 3] [4]};

%Time cputime
timeini = cputime

%All preferences
M = {Expert1, Expert2, Expert3, Expert4};

%Step 2: Standarize cost attributes

disp("Standarize cost attributes")
disp("============================")

for i=1: q
    for j=1: n
       for k=1: length(i_cost)
          M{i}{j, i_cost(k)} = flip(M_g(i) - M{i}{j, i_cost(k)});
       end
    end
end

%Matrix with standarized values
celldisp(M)


%Step 3: Unify preferences

%max granularity
max_g = max(M_g);
%scale max granularity
scale_gq = [0:max_g];
gq = length(scale_gq);

U = {};
for i=1: q %for all experts
    if M_g(i) ~= max_g
        scale_gp = [0:M_g(i)];
        gp = length(scale_gp);
        for j=1: n %for all alternatives
            for k=1: l %for all criteria
                hflts = M{i}{j, k}; %HFLTS
                hflts_mus = {};
                for t=1: length(hflts) %elements of HFLTS
                    mus = {};
                    m = hflts(t);
                    for r=1: gq %target scale lmin
                        if (scale_gq(r) ./ gq) <= (m ./ gp) && (m ./ gp) <= (scale_gq(r) + 1) ./ gq
                            lmin = scale_gq(r);
                            break;
                        end
                    end
                    for s=lmin+1: gq %target scale lmax
                        if (scale_gq(s) ./ gq) <= ((m + 1) ./ gp) && ((m + 1) ./ gp) <= (scale_gq(s) + 1) ./ gq
                            lmax = scale_gq(s);
                            break;
                        end
                    end
                    for lvalue=lmin: lmax
                        if lmin < lvalue && lvalue < lmax
                            mu = 1;
                        elseif lvalue == lmin
                            mu = (((lvalue + 1) ./ gq) - (m ./ gp)) * gq;
                        elseif lvalue == lmax
                            mu = (((m + 1) ./ gp) - (lvalue ./ gq)) * gq;
                        elseif lvalue == lmin && lmin == lmax
                            mu = gq ./ gp;
                        else
                            mu = 0;
                        end
                        mus = [mus [lvalue mu]];
                    end
                    hflts_mus = [hflts_mus mus];
                end
                U{i}{j, k} = hflts_mus;
            end
        end
    else %expert scale does not need to change
         for j=1: n %for all alternatives
            for k=1: l %for all criteria
                hflts = M{i}{j, k};
                mus = {};
                for t=1: length(hflts)
                    m = hflts(t);
                    mus = [mus [m 1]];
                end
                U{i}{j, k} = mus;
            end
        end
    end
end

disp("Unify values")
disp("============================")
%matriz with unified values for all the experts
celldisp(U)

%Step 4: Calculate the utility values

V = {};
for i=1: q %for all experts
    for j=1: n %for all alternatives
        for k=1: l %for all criteria
            h = U{i}{j, k};
            sum_mu = 0;
            for p=1: length(h)
                he = h{p};
                sum_mu = sum_mu + he(2);
            end
            v = 0;
            for p=1: length(h)
                he = h{p};
                v = v + (he(2) * he(1)) / sum_mu;
            end
            V{i}{j, k} = v;
        end
    end
end

disp("Utility values")
disp("============================")
%matrix with utility values
V

%Step 5: Calculate the u-rejoice and u-regret values

delta = 0.3;
Urej_input = {};
Urej_output = {};
for i=1: q %for all experts
    for j=1: n %for all alternatives
        for k=1: length(i_cost) %for cost criteria (inputs)
            Urej_input{i}{j, k} = V{i}{j, i_cost(k)} + (1 - exp(-delta * (max(cell2mat(V{i}(:, i_cost(k)))) - V{i}{j, i_cost(k)})));
        end
        for k=1: length(i_benf) %for benefit criteria (outputs)
            Urej_output{i}{j, k} = V{i}{j, i_benf(k)} + (1 - exp(-delta * (V{i}{j, i_benf(k)} - min(cell2mat(V{i}(:, i_benf(k)))))));
        end
    end
end

disp("U-rejoice values")
disp("============================")

disp("Inputs");
Urej_input
disp("Outputs");
Urej_output

Ureg_input = {};
Ureg_output = {};
for i=1: q %for all experts
    for j=1: n %for all alternatives
        for k=1: length(i_cost) %for cost criteria (inputs)
            Ureg_input{i}{j, k} = V{i}{j, i_cost(k)} + (1 - exp(-delta * (min(cell2mat(V{i}(:, i_cost(k)))) - V{i}{j, i_cost(k)})));
        end
        for k=1: length(i_benf) %for benefit criteria (outputs)
            Ureg_output{i}{j, k} = V{i}{j, i_benf(k)} + (1 - exp(-delta * (V{i}{j, i_benf(k)} - max(cell2mat(V{i}(:, i_benf(k)))))));
        end
    end
end

disp("U-regret values")
disp("============================")

disp("Inputs");
Ureg_input
disp("Outputs");
Ureg_output


%Step 6: Calculate the total utility values

gamma = 0.5;
U_inputs = {};
U_outputs = {};
for i=1: q %for all experts
    for j=1: n %for all alternatives
        for k=1: length(i_cost) %for benefit criteria (outputs)
            U_inputs{i}{j, k} = gamma * Urej_input{i}{j, k} + (1 - gamma) * Ureg_input{i}{j, k};
        end
        for k=1: length(i_benf)
            U_outputs{i}{j, k} = gamma * Urej_output{i}{j, k} + (1 - gamma) * Ureg_output{i}{j, k};
        end
    end
end

disp("Total utility values")
disp("============================")
disp("Inputs");
U_inputs
disp("Outputs");
U_outputs

%Step 7.1 Calculate the benevolent cross-efficiency

pkg load optim;
BE = {};
for ex=1: q %for all experts
    X = transpose(cell2mat(U_inputs{ex}));
    Y = transpose(cell2mat(U_outputs{ex}));
    n=size(X',1);m=size(X,1);s=size(Y,1);
    A=[-X' Y'];
    b=zeros(n,1);
    LB=zeros(m+s,1);UB=[];
    for i=1:n
        Aeq=[X(:,i)' zeros(1,s)];
        beq=1;
        f=transpose([zeros(1,m) -Y(:,i)']);
        w(:,i)=linprog(f,A,b,Aeq,beq,LB,UB);
        E(i,i)=Y(:,i)'*w(m+1:m+s,i);
        for k=1:n
            x=[sum(X')-X(:,i)'];
            y=[sum(Y')-Y(:,i)'];
            F=transpose([zeros(1,m) -y]);
            Aeq1=[x zeros(1,s);E(i,i)*X(:,i)' -Y(:,i)'];
            beq1=[1;0];
            vlin1(:,i)=linprog(F,A,b,Aeq1,beq1,LB,UB);
            E1B(i,k)=(Y(:,k)'*vlin1(m+1:m+s,i))/(X(:,k)'*vlin1(1:m,i));
        end
    end
    BE = [BE, E1B];
end

disp("Benevolent cross efficiency")
disp("============================")
BE

%Step 7.2 Calculate the aggressive cross-efficiency

AG = {};
for ex=1: q %for all experts
    X = transpose(cell2mat(U_inputs{ex}));
    Y = transpose(cell2mat(U_outputs{ex}));
    n=size(X',1);m=size(X,1);s=size(Y,1);
    A=[-X' Y'];
    b=zeros(n,1);
    LB=zeros(m+s,1);UB=[];
    for i=1:n
        Aeq=[X(:,i)' zeros(1,s)];
        beq=1;
        f=transpose([zeros(1,m) -Y(:,i)']);
        w(:,i)=linprog(f,A,b,Aeq,beq,LB,UB);
        E(i,i)=Y(:,i)'*w(m+1:m+s,i);
        for k=1:n
            x=[sum(X')-X(:,i)'];
            y=[sum(Y')-Y(:,i)'];
            F=transpose([zeros(1,m) y]);
            Aeq1=[x zeros(1,s);E(i,i)*X(:,i)' -Y(:,i)'];
            beq1=[1;0];
            vlin2(:,i)=linprog(F,A,b,Aeq1,beq1,LB,UB);
            E1A(i,k)=(Y(:,k)'*vlin2(m+1:m+s,i))/(X(:,k)'*vlin2(1:m,i));
        end
    end
    AG = [AG, E1A];
end

disp("Aggressive cross efficiency")
disp("============================")
AG

%Step 8: Determine the cross-efficiency intervals

LBounds = {};
low_p = zeros(n, q);
for i=1: q %for all experts
    lowerV = [];
    for j=1: n %for all alternatives
        LBounds{i}(j) = sum(AG{i}(j,:))/n;
    end
    low_p(:,i) = [transpose(cell2mat(LBounds(i)))];
end

UBounds = {};
up_p = zeros(n, q);
for i=1: q %for all experts
    upperV = [];
    for j=1: n %for all alternatives
        UBounds{i}(j) = sum(BE{i}(j,:))/n;
    end
    up_p(:,i) = [transpose(cell2mat(UBounds(i)))];
end

disp("Cross efficiency intervals")
disp("============================")
disp("Lower values");
low_p
disp("Upper values");
up_p

%Step 9: Stochastic DEA

number=200;%modify this value by hand
alpha=zeros(number,5);
for g=1:number
    sigma=zeros(5,4);
        for h=1:5
            for j=1:4
                sigma(h,j)=(up_p(h,j)-low_p(h,j)).*rand(1,1)+low_p(h,j);
            end
        end
    X=[0.2;0.2;0.2;0.2;0.2]';
    Y=sigma';
    m=1;
    s=4;
    n=5;
    for i = 1:n
        f = transpose([zeros(1,m) -Y(:,i)']);
        A = [-X' Y'];
        b = [zeros(n,1)];
        Aeq = [X(:,i)' zeros(1,s)];
        beq = [1];
        lb = [zeros(m+s,1)]; ub = [];
        w(:,i) = linprog(f,A,b,Aeq,beq,lb,ub);
        E(i,i) = Y(:,i)'* w(m+1:m+s,i);

    for k = 1:n
        F = transpose([zeros(1,m+s) -1]);
        A1 = [-X' Y' zeros(n,1);zeros(1,m) -Y(1,i)' 0 0 0 1;zeros(1,m) 0 -Y(2,i)' 0 0 1;zeros(1,m) 0 0 -Y(3,i)' 0 1;zeros(1,m) 0 0 0 -Y(4,i)' 1];
        b1 = [zeros(n,1);0;0;0;0];
        Aeq1 = [X(:,i)' zeros(1,s) 0;zeros(1,m) Y(:,i)' 0];
        beq1 = [1;E(i,i)];
        LB = [zeros(m+s+1,1)];UB=[];
        vdea(:,i) = linprog(F,A1,b1,Aeq1,beq1,LB,UB);
        H(i,k)=(Y(:,k)'*vdea(m+1:m+s,i))/(X(:,k)'*vdea(1:m,i));
        avg_Ej = mean(H);
    end
    end
    alpha(g,:)=avg_Ej;
end

alpha_sort=sort(alpha,2);
alpha_average=sum(alpha)./number

[~,p] = sort(alpha_average,'descend');
r = 1:length(alpha_average);
r(p) = r

printf('Total cpu time: %f seconds\n', cputime-timeini);
