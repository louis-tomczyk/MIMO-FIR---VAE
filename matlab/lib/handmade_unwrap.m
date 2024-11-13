function y = handmade_unwrap(x,varargin)

    if ~isempty(varargin)
      ref       = 90/varargin{1};
    else
      ref       = 90
    end

    xdiff       = diff(x);
    index_jumps = find(abs(xdiff)>ref);
    njumps      = length(index_jumps);

    count       = 0;
    y           = x;

    while count < njumps
        count       = count+1;
        k           = index_jumps(count);
        offset      = -2*ref*sign(x(k+1)-x(k));
        if count == 1
            y(k+1:end)  = x(index_jumps(count)+1:end)+offset;
        else
            y(k+1:end)  = y(index_jumps(count)+1:end)+offset;
        end
    end
