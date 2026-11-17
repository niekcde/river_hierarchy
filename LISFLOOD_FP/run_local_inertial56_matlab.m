function run_local_inertial56_matlab(folder)
% CPU MATLAB pilot matching local_inertial_pilot56.py.
% Prepared but NOT executed: MATLAB license checkout failed in this session.
% Equations/interface rules adapted from LISFLOOD_FP_routing_GPU_2.m.
% Corrected time accounting, exact fixed stage, linear forcing, true outlet Q.
if nargin==0
    folder=fullfile(fileparts(mfilename('fullpath')),'results','example56','local_inertial_pilot');
end
for name={'base','S014_unit_10'}
    state=name{1};load(fullfile(folder,[state '_matlab_input.mat']),'m','times','ft','fq');
    for refinement=1:2
        cfl=.7/refinement;max_dt=300/refinement;h=[];q=[];
        for day=1:60
            [~,~,h,q,audit]=route(m,[0;86400],[0;86400],[fq(1);fq(1)],h,q,max_dt,cfl);
            if audit.final_node_imbalance_cms/fq(1)<1e-7, break; end
        end
        assert(day<60,'Spinup failed');
        [wave,depth,h,q,audit]=route(m,times,ft,fq,h,q,max_dt,cfl);
        tag={'standard','refined'};prefix=fullfile(folder,[state '_matlab_' tag{refinement}]);
        names={'time_seconds','outlet_cms','storage_m3','inflow_volume_m3','outflow_volume_m3','clipping_volume_m3','mass_residual_m3'};
        writetable(array2table(wave,'VariableNames',names),[prefix '.csv']);
        save([prefix '.mat'],'wave','depth','h','q','audit','day');
        disp(audit);
    end
end
end

function [wave,depth,h,q,audit]=route(m,times,ft,fq,h,q,max_dt,cfl)
g=9.80665;N=numel(m.area);a=m.a;b=m.b;terminal=b==N+1;
if isempty(h),h=zeros(N+1,1);q=zeros(numel(a),1);end
h(end)=m.ghost_depth;initial_storage=m.area'*h(1:N);
vin=0;vout=0;clipping=0;t=times(1);steps=0;limited=0;
max_depth=max(h(1:N));min_depth=min(h(1:N));
wave=zeros(numel(times),7);depth=zeros(numel(times),N);
wave(1,:)=[t,sum(q(terminal).*m.wf(terminal)),initial_storage,0,0,0,0];depth(1,:)=h(1:N)';
for output=2:numel(times)
    stop=times(output);
    while t<stop-1e-8
        k=find(ft>t+1e-8,1);if isempty(k),knot=Inf;else,knot=ft(k);end
        dt=min([max_dt,cfl*min(m.dx./sqrt(g*max(h(1:N),eps))),stop-t,knot-t]);
        assert(dt>0);
        eta=m.z+h;
        hf=max(max(eta(a),eta(b))-max(m.z(a),m.z(b)),eps);
        slope=(eta(b)-eta(a))./m.dxf;
        newq=(q-g*dt*hf.*slope)./(1+g*dt*m.n^2*abs(q)./hf.^(7/3));
        limit=hf.*sqrt(g*hf);limited=limited+nnz(abs(newq)>limit);
        q=sign(newq).*min(abs(newq),limit);Q=q.*m.wf;
        inlet=interp1(ft,fq,min(max(t+dt/2,ft(1)),ft(end)),'linear');
        balance=accumarray(b,Q,[N+1,1])-accumarray(a,Q,[N+1,1]);
        balance(m.source)=balance(m.source)+inlet;
        trial=h(1:N)+balance(1:N)*dt./m.area;
        correction=max(eps-trial,0);clipping=clipping+m.area'*correction;
        h(1:N)=trial+correction;h(end)=m.ghost_depth;
        vin=vin+inlet*dt;vout=vout+sum(Q(terminal))*dt;t=t+dt;steps=steps+1;
        max_depth=max(max_depth,max(h(1:N)));min_depth=min(min_depth,min(h(1:N)));
    end
    t=stop;storage=m.area'*h(1:N);
    wave(output,:)=[t,sum(q(terminal).*m.wf(terminal)),storage,vin,vout,clipping,storage-initial_storage-vin+vout-clipping];
    depth(output,:)=h(1:N)';
end
Q=q.*m.wf;balance=accumarray(b,Q,[N+1,1])-accumarray(a,Q,[N+1,1]);
balance(m.source)=balance(m.source)+interp1(ft,fq,min(max(t,ft(1)),ft(end)),'linear');
audit=struct('steps',steps,'limited_face_steps',limited,'clipping_volume_m3',clipping,...
    'maximum_depth_m',max_depth,'minimum_depth_m',min_depth,...
    'maximum_mass_residual_m3',max(abs(wave(:,7))),...
    'final_node_imbalance_cms',max(abs(balance(1:N))));
end
