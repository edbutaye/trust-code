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
// File:        Source_Robin.h
// Directory:   $TRUST_ROOT/src/VEF/Turbulence
// Version:     Version:
//
//////////////////////////////////////////////////////////////////////////////

#ifndef Source_Robin_included
#define Source_Robin_included

#include <Source_base.h>
#include <Ref_Zone_VEF.h>
#include <Ref_Zone_Cl_VEF.h>

class Convection_Diffusion_std;

class Source_Robin : public Source_base
{
  Declare_instanciable(Source_Robin);

public:
  virtual void associer_pb(const Probleme_base&);
  DoubleTab& ajouter(DoubleTab&) const;
  DoubleTab& calculer(DoubleTab&) const;
  void mettre_a_jour(double) { };
  void completer();
  inline const DoubleVect& tab_u_star() const
  {
    return tab_u_star_;
  };

protected:
  REF(Zone_VEF) la_zone_VEF;
  REF(Zone_Cl_VEF) la_zone_Cl_VEF;
  void associer_zones(const Zone_dis&, const Zone_Cl_dis&);
//  double dt_post;
  Noms noms_parois;
  mutable DoubleVect tab_u_star_;	// mutable pour etre modifie dans ajouter() => pas top, devrait etre modifie dans mettre_a_jour
};

#endif
