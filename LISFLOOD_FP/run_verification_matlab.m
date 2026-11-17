function run_verification_matlab(folder)
% CPU normal-flow adaptation of supplied MATLAB local-inertial formulation.
% Uses held forcing, correct step accounting, true boundary flux, and normal
% outlet. Python-prepared geometry and steady initial state are shared.
% This checks language equivalence, not the unmodified supplied full program.
D=load(fullfile(folder,'input.mat'));
z=D.z;w=D.w;dx=D.dx;a=D.a;b=D.b;wf=D.wf;dxf=D.dxf;
source=D.source;monitor_face=D.monitor_face;sample_indices=D.sample_indices;
times=D.times;forcing=D.forcing;h=D.h;q=D.q;cfl=D.cfl;max_dt=D.max_dt;
n=D.n;outlet_slope=D.outlet_slope;stage=D.stage;normal=D.normal;
a=a+1;b=b+1;source=source+1;monitor_face=monitor_face+1;
sample_indices=sample_indices+1;N=numel(w);M=numel(a);area=w.*dx;
h=h(:);q=q(:);terminal=b==N+1;interior=~terminal;g=9.80665;
wave=zeros(numel(times),9);depth=zeros(numel(sample_indices),N);flow=zeros(numel(sample_indices),M);
hmax=h;hmin=h;initial=sum(area.*h);vin=0;vout=0;clip=0;vm=0;steps=0;limited=0;
minQ=Inf;maxFr=0;dtmin=Inf;dtmax=0;
for it=1:numel(times)
    if it>1
        t=times(it-1);stop=times(it);inlet=forcing(it-1);
        while t<stop-1e-8
            dt=min([max_dt,stop-t,cfl*min(dx./sqrt(g*max(h,1e-12)))]);
            eta=z(1:N)+h;eta=[eta;stage];
            hf=max(max(eta(a),eta(b))-max(z(a),z(b)),1e-12);
            slope=(eta(b)-eta(a))./dxf;
            newq=(q-g*dt*hf.*slope)./(1+g*dt*n^2*abs(q)./hf.^(7/3));
            limit=hf.*sqrt(g*hf);
            if normal
                hf(terminal)=max(h(a(terminal)),1e-12);
                newq(terminal)=max(h(a(terminal)),0).^(5/3)*sqrt(outlet_slope)/n;
                limited=limited+nnz(abs(newq(interior))>limit(interior));
                newq(interior)=max(-limit(interior),min(limit(interior),newq(interior)));
            else
                limited=limited+nnz(abs(newq)>limit);
                newq=max(-limit,min(limit,newq));
            end
            q=newq;Q=q.*wf;
            balance=accumarray(b,Q,[N+1,1])-accumarray(a,Q,[N+1,1]);balance(source)=balance(source)+inlet;
            trial=h+dt*balance(1:N)./area;clip=clip+sum(max(-trial,0).*area);h=max(trial,0);
            hmax=max(hmax,h);hmin=min(hmin,h);
            vin=vin+inlet*dt;vout=vout+sum(Q(terminal))*dt;vm=vm+Q(monitor_face)*dt;
            minQ=min(minQ,min(Q));maxFr=max(maxFr,max(abs(q)./(hf.*sqrt(g*hf))));
            t=t+dt;steps=steps+1;dtmin=min(dtmin,dt);dtmax=max(dtmax,dt);
        end
    end
    storage=sum(area.*h);out=sum(q(terminal).*wf(terminal));residual=storage-initial-vin+vout-clip;
    wave(it,:)=[times(it),out,storage,vin,vout,clip,residual,q(monitor_face)*wf(monitor_face),vm];
    k=find(sample_indices==it);depth(k,:)=repmat(h',numel(k),1);flow(k,:)=repmat((q.*wf)',numel(k),1);
end
balance=accumarray(b,q.*wf,[N+1,1])-accumarray(a,q.*wf,[N+1,1]);balance(source)=balance(source)+forcing(end);
audit=[steps,limited,clip,max(abs(wave(:,7))),max(hmax),min(hmin),minQ,dtmin,dtmax,max(abs(balance(1:N))),maxFr];
save(fullfile(folder,'matlab_result.mat'),'wave','depth','flow','h','q','hmax','hmin','audit','-v7');
disp(folder);disp(audit);
end
