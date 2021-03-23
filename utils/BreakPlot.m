function [figure]=BreakPlot(x,y,y_break_start,y_break_end,break_type, ...
                newMarkers, newColors, lineWidth, legendStrs, figure)
% BreakPlot(x,y,y_break_start,y_break_end,break_type)
% Produces a plot who's y-axis skips to avoid unecessary blank space
% 
% INPUT
% x
% y
% y_break_start
% y_break_end
% break_type
%    if break_type='RPatch' the plot will look torn
%       in the broken space
%    if break_type='Patch' the plot will have a more
%       regular, zig-zag tear
%    if break_plot='Line' the plot will merely have
%       some hash marks on the y-axis to denote the
%       break
%
% USAGE:
% figure;
% BreakPlot(rand(1,21),[1:10,40:50],10,40,'Line');
% figure;
% BreakPlot(rand(1,21),[1:10,40:50],10,40,'Patch');
% figure;
% BreakPlot(rand(1,21),[1:10,40:50],10,40,'RPatch');
% figure;
% x=rand(1,21);y=[1:10,40:50];
% subplot(2,1,1);plot(x(y>=40),y(y>=40),'.');
% set(gca,'XTickLabel',[]);
% subplot(2,1,2);plot(x(y<=20),y(y<=20),'.');
%
% IT'S NOT FANCY, BUT IT WORKS.

% Michael Robbins
% robbins@bloomberg.net
% michael.robbins@bloomberg.net

% data
if nargin<5 break_type='RPatch'; end;
if nargin<4 y_break_end=40; end;
if nargin<3 y_break_start=10; end;
if nargin<2 y=[1:10,40:50]; end;
if nargin<1 x=rand(1,21); end;

N = size(y, 1);

y_break_mid=(y_break_end-y_break_start);

% erase useless data
% x(y>y_break_start & y <y_break_end)=[];
% y(y>y_break_start & y <y_break_end)=[];

% leave room for the y_break_end
y0=y;x0=x;
x=[];y=[];
for id = 1: N
    [junk,i]=min(y0(id,:)>=y_break_end);
    if i>y_break_end && max(y0(id,:)) > y_break_end
        x=[x; x0(id, 1:i-1) NaN x0(id, i:end)];
        y=[y; y0(id, 1:i-1) y_break_mid y0(id, i:end)];
    else
        x=[x; x0(id,:) NaN];
        y=[y; y0(id,:) y_break_mid];
    end;
end

% remap
y2=y;
y2(y2>=y_break_end)=y2(y2>=y_break_end)-y_break_mid;

% plot
figure , hold on
for id = 1: N
plot(x(id,:),y2(id,:), newMarkers{id}, 'LineWidth', lineWidth,...
                'Color', newColors(id,:),...
                'MarkerSize', 12,...
                'MarkerFaceColor', 'auto');
end

% make break
xlim=get(gca,'xlim');
ytick=get(gca,'YTick');
[junk,i]=min(ytick<=y_break_start);
% y=(ytick(i)-ytick(i-1))./2+ytick(i-1);
y=ytick(i-1);
dy=(ytick(2)-ytick(1))./10;
xtick=get(gca,'XTick');
x=xtick(1);
dx=(xtick(2)-xtick(1))./10;

switch break_type
    case 'Patch',
		% this can be vectorized
        dx=(xlim(2)-xlim(1))./10;
        yy=repmat([y-2.*dy y-dy],1,6);
        xx=xlim(1)+dx.*[0:11];
		patch([xx(:);flipud(xx(:))], ...
            [yy(:);flipud(yy(:)-2.*dy)], ...
            [.8 .8 .8])
    case 'RPatch',
		% this can be vectorized
        dx=(xlim(2)-xlim(1))./100;
        yy=y+rand(101,1).*2.*dy;
        xx=xlim(1)+dx.*(0:100);
		patch([xx(:);flipud(xx(:))], ...
            [yy(:);flipud(yy(:)-2.*dy)], ...
            [.8 .8 .8])
    case 'Line',
		line([x-dx x   ],[y-2.*dy y-dy   ], 'Color', 'k', 'LineWidth', 4);
		line([x    x+dx],[y+dy    y+2.*dy], 'Color', 'k', 'LineWidth', 4);
		line([x-dx x   ],[y-3.*dy y-2.*dy], 'Color', 'k', 'LineWidth', 4);
		line([x    x+dx],[y+2.*dy y+3.*dy], 'Color', 'k', 'LineWidth', 4);
end;
set(gca,'xlim',xlim);

% map back
ytick(ytick>y_break_start)=ytick(ytick>y_break_start)+y_break_mid;
for i=1:length(ytick)
   yticklabel{i}=sprintf('%.2f',ytick(i));
end;
set(gca,'yticklabel',yticklabel);
legend(legendStrs);
hold off;

