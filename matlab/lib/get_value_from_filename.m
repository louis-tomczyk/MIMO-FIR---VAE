function out = get_value_from_filename(folderPath,quantity,varargin)

% ---------------------------------------------
% ----- INFORMATIONS -----
%   Function name   : GET_VALUE_FROM_FILENAME
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   ArXivs          : 2023-03-06 - creation (1.0.0)
%   Date            : 2024-03-05 - self-sufficiency + variable number of files as input
%   Version         : 1.1.0
%   License         : cc-by-nc-sa
%                       CAN:    modify - distribute
%                       CANNOT: commercial use
%                       MUST:   share alike - include license
%
% ----- MAIN IDEA -----
%   Get a value contained in the name of a file
%
% ----- INPUTS -----
%   FOLDERPATH  (string)        Folder that contain the files
%   QUANTITY    (string)        Keyword before/after which is located the figure
%   VARARGIN    (cell array)    Filenames to proceed
%
% ----- OUTPUTS -----
%   OUT         (scalar)        numeric value extracted from the name of the file
%
% ----- BIBLIOGRAPHY -----
% ----------------------------------------------

cd(folderPath)

if nargin == 2
    nfiles          = length(dir(pwd))-2;
    folder_struct   = dir(pwd);
    out             = zeros(nfiles,1);

    for k=1:nfiles
        filename    = folder_struct(k+2).name;
        out(k)      = get_number_from_string_in(filename,quantity);
    end

else
    nfiles          = length(varargin);
    out             = zeros(nfiles,1);
    for k=1:nfiles
        out(k)      = get_number_from_string_in(varargin{1}{k},quantity);
    end
end

out = sort(out);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NESTED FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------------------------------
% ----- CONTENTS -----
%   get_number_from_string_in   stringIn,what,varargin          out
% ---------------------------------------------

function out = get_number_from_string_in(stringIn,what,varargin)

iwhat       = strfind(stringIn,what);
if nargin == 2
    iendwhat    = iwhat+length(what);
    idashes     = strfind(stringIn,'-');
    [~,itmp]    = max(idashes-iendwhat>0);
    idashNext   = idashes(itmp);
    strTmp      = stringIn(iendwhat+1:idashNext-1);
else
    if nargin > 2
        if iwhat-varargin{1}<1
            istart = 1;
        else
            istart = iwhat-varargin{1};
        end
        if nargin == 4
            if iwhat+varargin{2}>length(stringIn)
                iend = length(stringIn);
            else
                iend = iwhat+varargin{2};
            end
        end
        strTmp  = stringIn(istart:iend);
    end
end

indexes = regexp(strTmp,'[0123456789.]');
out     = str2double(strTmp(indexes));

        