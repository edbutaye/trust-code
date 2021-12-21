/****************************************************************************
* Copyright (c) 2022, CEA
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
// File:        Champ_Front_xyz_Tabule.h
// Directory:   $TRUST_ROOT/src/Kernel/Champs
// Version:     1
//
//////////////////////////////////////////////////////////////////////////////

#ifndef Champ_Front_xyz_Tabule_included
#define Champ_Front_xyz_Tabule_included

#include <Champ_front_txyz.h>
#include <Table.h>

/////////////////////////////////////////////////////////////////////////////
//
// .DESCRIPTION : class Champ_Front_xyz_Tabule
//
// <Description of class Champ_Front_xyz_Tabule>
//
/////////////////////////////////////////////////////////////////////////////

class Champ_Front_xyz_Tabule : public Champ_front_txyz
{

  Declare_instanciable( Champ_Front_xyz_Tabule ) ;

public :

  void mettre_a_jour(double temps);
  double valeur_au_temps_et_au_point(double temps,int som,double x,double y, double z,int comp) const;

protected :
  Table la_table;

};
#endif /* Champ_Front_xyz_Tabule_included */
