/****************************************************************************
* Copyright (c) 2015 - 2016, CEA
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
// File:        Terme_Source_Coriolis_VDF_Face.h
// Directory:   $TRUST_ROOT/src/VDF/Sources
// Version:     /main/10
//
//////////////////////////////////////////////////////////////////////////////


#ifndef Terme_Source_Coriolis_VDF_Face_included
#define Terme_Source_Coriolis_VDF_Face_included



//
// .DESCRIPTION class Terme_Source_Coriolis_VDF_Face
//  Cette classe permet de calculer la force de Coriolis en VDF
//
//
// .SECTION voir aussi
// Terme_Source_Coriolis

#include <Source_base.h>
#include <Terme_Source_Coriolis.h>
#include <Ref_Zone_VDF.h>
#include <Ref_Zone_Cl_VDF.h>
#include <DoubleTab.h>

class Probleme_base;
class Navier_Stokes_std;

class Terme_Source_Coriolis_VDF_Face : public Source_base, public Terme_Source_Coriolis
{
  Declare_instanciable(Terme_Source_Coriolis_VDF_Face);

public :
  void associer_pb(const Probleme_base& );
  DoubleTab& ajouter(DoubleTab& ) const;
  DoubleTab& calculer(DoubleTab& ) const;
  inline const DoubleVect& omega() const ;
  void mettre_a_jour(double temps)
  {
    Terme_Source_Coriolis::mettre_a_jour(temps);
  }

protected :

  REF(Zone_VDF) la_zone_VDF;
  REF(Zone_Cl_VDF) la_zone_Cl_VDF;
  DoubleTab la_source;
  void associer_zones(const Zone_dis& ,const Zone_Cl_dis& );
  void calculer_force_de_Coriolis() const;

};

// Ajoute pour compatibilite avec Quasi-Compressible
class Terme_Source_Coriolis_QC_VDF_Face : public Terme_Source_Coriolis_VDF_Face
{
  Declare_instanciable(Terme_Source_Coriolis_QC_VDF_Face);
};


// Fonctions inline de la classe Terme_Source_Coriolis_VDF_Face

inline const DoubleVect& Terme_Source_Coriolis_VDF_Face::omega() const
{
  return Terme_Source_Coriolis::omega();
}

#endif
