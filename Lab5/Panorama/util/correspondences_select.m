%function [X1,X2]=correspondences_select(im1,im2,mflag)
%
% Select corresponding points by clicking. Points are
% automatically partitioned into left and right sets (or top 
% and bottom sets if in landscape mode).
%
% Left click to select a point.
% Middle click to cancel last click.
% Right click to end.
%
% IM1    Left/Top  image
% IM2    Right/Bottom image
% MFLAG  Mode: 1 for landscape (default), 0 for portrait
%
%Per-Erik Forssen, June 2008
function [X1,X2]=correspondences_select(im1,im2,mflag)

if nargin<3,mflag=1;end

marker_size=20;
button_left=1;
button_mid=2;
button_right=3;
button_esc=27;

size1=size(im1);
size2=size(im2);
if(any(size1-size2)~=0)
    error('The two images have different sizes.');
end
rows=size1(1);
cols=size1(2);

clf
if mflag,
    image([im1 im2]);
else
    image([im1;im2]);
end
axis image
hold on

X1=[];
X2=[];
cind=1;

hlist=[]; % Handles to all lines [3xN]
select_points=1;
while(select_points)
    % Select first point in correspondence
    [x1,y1,b]=ginput(1);
    h=plot(x1,y1,'gx');
    set(h,'MarkerSize',marker_size);
    switch(b)
        case button_mid,
            delete(h);
            if(cind>1)
                cind=cind-1;
                delete(hlist(:,cind));
                X1=X1(:,1:cind-1);
                X2=X2(:,1:cind-1);
                hlist=hlist(:,1:cind-1);
            end
            continue;
        case {button_right,button_esc},
            delete(h);
            select_points=0;
            break;
    end
    % Select second point in correspondence
    [x2,y2,b]=ginput(1);
    h=[h;zeros(2,1)];
    h(2)=plot(x2,y2,'gx');
    set(h(2),'MarkerSize',marker_size);
    h(3)=plot([x1 x2],[y1 y2],'g-');
    switch(b)
        case button_left,
            if mflag,
                % Landscape
                if x2>x1,
                    p1=[x1;y1];p2=[x2-cols;y2];
                else
                    p1=[x2;y2];p2=[x1-cols;y1];
                end
            else
                % Portrait
                if y2>y1;
                    p1=[x1;y1];p2=[x2;y2-rows];
                else
                    p1=[x2;y2];p2=[x1;y1-rows];
                end
            end
            X1=[X1 p1];
            X2=[X2 p2];
            hlist=[hlist h];
            cind=cind+1;
        case button_mid,
            delete(h);
            continue;
        case {button_right,button_esc},
            delete(h);
            select_points=0;
            break;
    end    
end

X1=X1(:,1:cind-1);
X2=X2(:,1:cind-1);
