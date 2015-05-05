/****************************************************************************
* Copyright (c) 2015, CEA
* All rights reserved.
* 
* Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
* 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
* IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
* OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
*****************************************************************************/
//////////////////////////////////////////////////////////////////////////////
//
// File:        EcrFicCollecteBin.h
// Directory:   $TRUST_ROOT/src/Kernel/Utilitaires
// Version:     /main/11
//
//////////////////////////////////////////////////////////////////////////////


#ifndef EcrFicCollecteBin_included
#define EcrFicCollecteBin_included

#include <EcrFicCollecte.h>


class Objet_U;



//////////////////////////////////////////////////////////////////////////////
//
// .DESCRIPTION
//    Ecriture dans un fichier au format binaire
//    Cette classe implemente les operateurs et les methodes virtuelles de la clase SFichier de la facon suivante :
//    Il y a autant de fichiers que de processus, physiquement localises sur le disque de la machine hebergeant la tache maitre de l'application Trio-U (le processus de rang 0 dans le groupe "tous")
//    Le processus maitre recoit chaque item a ecrire de chacun des processus et dans chacun chacun des fichiers.
//    il en est de meme pour les methodes d'inspection de l'etat d'un fichier.
// .SECTION voir aussi
//////////////////////////////////////////////////////////////////////////////

class EcrFicCollecteBin : public EcrFicCollecte
{
  Declare_instanciable_sans_constructeur_ni_destructeur(EcrFicCollecteBin);
  // le maitre collecte l'information de tous les PE et l'ecrit dans differents fichiers
public:
  EcrFicCollecteBin()
  {
    set_bin(1);
  };
  EcrFicCollecteBin(const char* name,IOS_OPEN_MODE mode=ios::out)
  {
    set_bin(1);
    ouvrir(name,mode);
  };
  ~EcrFicCollecteBin()
  {
    close();
  };
};



#endif
