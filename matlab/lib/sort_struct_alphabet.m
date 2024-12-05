% ---------------------------------------------
% ----- INFORMATIONS -----
%   Author          : louis tomczyk
%   Institution     : Telecom Paris
%   Email           : louis.tomczyk@telecom-paris.fr
%   Version         : 2.0.0
%   Date            : 2024-02-19
%   ArXivs          : 2023-08-30 (1.0.0)
%                   : 2024-02-19 (2.0.0) first structures, then others
%
% ----- Main idea ----- 
% ----- INPUTS -----
%   myStruct        (structure) the structure to sort
%
% ----- OUTPUTS -----
%   myStruct        (structure) the structure sorted
%
% ----- BIBLIOGRAPHY -----
% ---------------------------------------------

function myStruct = sort_struct_alphabet(myStruct)

    nfields         = length(fieldnames(myStruct));
    index_structs   = zeros(1,nfields);
    fieldNames      = fieldnames(myStruct);

    for k = 1:nfields
        if isstruct(myStruct.(fieldNames{k}))
            index_structs(k) = k;
        end
    end

    index_notStruct = find(index_structs == 0);
    index_structs   = index_structs(index_structs~=0);
    Nstructs        = length(index_structs);
    NnotStruct      = length(index_notStruct);
    structFields    = strings(1,Nstructs);
    otherFields     = strings(1,NnotStruct);

    for k = 1:Nstructs
        structFields(k) = fieldNames{index_structs(k)};
    end

    for k = 1:NnotStruct
        otherFields(k) = fieldNames{index_notStruct(k)};
    end

    structFields    = sort(structFields);
    otherFields     = sort(otherFields);
    sortedFields    = [structFields, otherFields];
    myStruct        = orderfields(myStruct, sortedFields);
end
