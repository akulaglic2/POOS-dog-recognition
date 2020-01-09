function [result,w,U,S,V,threshold]=dc_trainer(dog0,non_dog0,feature)
    nd=length(dog0(1,:));
    nc=length(non_dog0(1,:));
    [U,S,V] = svd([dog0,non_dog0],0); % reduced SVD
    animals = S*V';
    U = U(:,1:feature);
    dogs = animals(1:feature,1:nd);
    non_dogs = animals(1:feature,nd+1:nd+nc);
    
    md = mean(dogs,2);
    mc = mean(non_dogs,2);
    
    Sw=0; % within class variances
    for i=1:nd
        Sw = Sw + (dogs(:,i)-md)*(dogs(:,i)-md)';
    end
    for i=1:nc
        Sw = Sw + (non_dogs(:,i)-mc)*(non_dogs(:,i)-mc)';
    end
    
    Sb = (md-mc)*(md-mc)'; % between class
    [V2,D] = eig(Sb,Sw); % linear discriminant analysis
    [lambda,ind] = max(abs(diag(D)));
    w = V2(:,ind); w = w/norm(w,2);
    vdog = w'*dogs; vnon_dog = w'*non_dogs;
    result = [vdog,vnon_dog];
    
    if mean(vdog)>mean(vnon_dog)
        w = -w;
        vdog = -vdog;
        vnon_dog = -vnon_dog;
    end
    sortdog = sort(vdog);
    sortnon_dog = sort(vnon_dog);
    t1 = length(sortdog);
    t2 = 1;
    while sortdog(t1)>sortnon_dog(t2)
        t1 = t1-1;
        t2 = t2+1;
    end
    threshold = (sortdog(t1)+sortnon_dog(t2))/2;
end