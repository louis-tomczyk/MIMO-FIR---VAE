# %%
# ---------------------------------------------
# ----- INFORMATIONS -----
#   Author          : louis tomczyk
#   Institution     : Telecom Paris
#   Email           : louis.tomczyk@telecom-paris.fr
#   Arxivs          :
#   Date            : 2023-03-04
#   Version         : 1.0.0
#   Licence         : cc-by-nc-sa
#                     Attribution - Non-Commercial - Share Alike 4.0 International
# 
# ----- Main idea -----
# ----- INPUTS -----
# ----- BIBLIOGRAPHY -----
#   Articles/Books
#   Authors             : 
#   Title               :
#   Jounal/Editor       : 
#   Volume - N°         : 
#   Date                :
#   DOI/ISBN            :
#   Pages               :
#  ----------------------
#   Functions           : 
#   Author              : 
#   Author contact      : 
#   Affiliation         :
#   Date                : 
#   Title of program    : 
#   Code version        : 
#   Type                :
#   Web Address         :
# ---------------------------------------------
# %%

def optimise(loss,optimiser):

    #1/ gradient calculation
    #2/ weights update
    #3/ gradient clearing

    loss.backward()           #1
    optimiser.step()          #2
    optimiser.zero_grad()     #3