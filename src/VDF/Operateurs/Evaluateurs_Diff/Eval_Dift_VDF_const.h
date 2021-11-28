/****************************************************************************
* Copyright (c) 2021, CEA
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
// File:        Eval_Dift_VDF_const.h
// Directory:   $TRUST_ROOT/src/VDF/Operateurs/Evaluateurs_Diff
// Version:     /main/6
//
//////////////////////////////////////////////////////////////////////////////

#ifndef Eval_Dift_VDF_const_included
#define Eval_Dift_VDF_const_included

#include <Eval_Diff_VDF_const.h>
#include <Eval_Turbulence.h>
#include <Ref_Champ_Fonc.h>
#include <Champ_Fonc.h>
//
// .DESCRIPTION class Eval_Dift_VDF_const
//
// Cette classe represente un evaluateur de flux diffusif
// total (diffusion laminaire et turbulente)
// avec une diffusivite constante en espace.

//.SECTION
// voir aussi Eval_Diff_VDF_const

class Eval_Dift_VDF_const : public Eval_Diff_VDF_const, public Eval_Turbulence
{
public:
  inline void mettre_a_jour();
  inline void associer_diff_turb(const Champ_Fonc&);
  inline const Champ_Fonc& diffusivite_turbulente() const { return diffusivite_turbulente_.valeur(); }

  // Overloaded methods used by the flux computation in template class:
  inline double nu_1_impl(int i, int compo) const
  {
    const double nu_lam = Eval_Diff_VDF_const::nu_1_impl(i,compo);
    const double nu_turb = dv_diffusivite_turbulente(i);
    return nu_lam+nu_turb;
  }

  inline double nu_2_impl(int i, int compo) const { return Eval_Diff_VDF_const::nu_2_impl(i,compo); }
  inline double nu_t_impl(int i, int compo) const { return dv_diffusivite_turbulente(i);  }

  inline double nu_1_impl_face(int i, int j, int compo) const
  {
    return 0.5*(dv_diffusivite_turbulente(i)+dv_diffusivite_turbulente(j));
  }

  inline double nu_2_impl_face(int i, int j, int k, int l, int compo) const
  {
    return 0.25*(dv_diffusivite_turbulente(i)+dv_diffusivite_turbulente(j)+
                 dv_diffusivite_turbulente(k)+dv_diffusivite_turbulente(l));
  }

  inline double nu_lam_impl_face(int i, int j, int k, int l, int compo) const
  {
    return Eval_Diff_VDF_const::nu_2_impl_face(i,j,k,l,compo);
  }

  inline double nu_lam_impl_face2(int i, int j, int compo) const
  {
    return Eval_Diff_VDF_const::nu_1_impl_face(i,j,compo);
  }

  inline double compute_heq_impl(double d0, int i, double d1, int j, int compo) const
  {
    const double heq_lam = Eval_Diff_VDF_const::compute_heq_impl(d0, i, d1, j, compo);
    const double heq_turb= 0.5*(dv_diffusivite_turbulente(i)+dv_diffusivite_turbulente(j))/(d1+d0);
    return heq_lam + heq_turb;
  }

  inline virtual double get_equivalent_distance(int boundary_index,int local_face) const
  {
    return equivalent_distance[boundary_index](local_face);
  }

protected:
  REF(Champ_Fonc) diffusivite_turbulente_;
  DoubleVect dv_diffusivite_turbulente;
};

inline void Eval_Dift_VDF_const::mettre_a_jour()
{
  Eval_Diff_VDF_const::mettre_a_jour();
  update_equivalent_distance();  // from Eval_Turbulence
}

inline void Eval_Dift_VDF_const::associer_diff_turb(const Champ_Fonc& diff_turb)
{
  diffusivite_turbulente_ = diff_turb;
  dv_diffusivite_turbulente.ref(diff_turb.valeurs());
}

#endif /* Eval_Dift_VDF_const_included */
