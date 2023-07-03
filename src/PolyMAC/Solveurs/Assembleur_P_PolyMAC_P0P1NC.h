/****************************************************************************
* Copyright (c) 2023, CEA
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

#ifndef Assembleur_P_PolyMAC_P0P1NC_included
#define Assembleur_P_PolyMAC_P0P1NC_included

#include <Matrice_Morse_Sym.h>
#include <Assembleur_base.h>
#include <Domaine_PolyMAC_P0P1NC.h>
#include <TRUST_Ref.h>

class Domaine_Cl_PolyMAC;

class Assembleur_P_PolyMAC_P0P1NC : public Assembleur_base
{
  Declare_instanciable(Assembleur_P_PolyMAC_P0P1NC);

public:
  void associer_domaine_dis_base(const Domaine_dis_base& ) override;
  void associer_domaine_cl_dis_base(const Domaine_Cl_dis_base& ) override;
  const Domaine_dis_base& domaine_dis_base() const override;
  const Domaine_Cl_dis_base& domaine_Cl_dis_base() const override;
  int assembler(Matrice&) override;
  int assembler_rho_variable(Matrice&, const Champ_Don_base& rho) override;
  int assembler_QC(const DoubleTab&, Matrice&) override;
  int assembler_mat(Matrice&,const DoubleVect&,int incr_pression,int resoudre_en_u) override;
  void dimensionner_continuite(matrices_t matrices, int aux_only = 0) const override;
  void assembler_continuite(matrices_t matrices, DoubleTab& secmem, int aux_only = 0) const override;
  DoubleTab norme_continuite() const override;
  int modifier_secmem(DoubleTab&) override
  {
    return 1;
  }
  /* prise en compte des variations de pression aux CLs lors du calcul d'increments de pression.
     fac est le coefficient tel que p_final - press = fac * sol */
  void modifier_secmem_pour_incr_p(const DoubleTab& press, const double fac, DoubleTab& incr) const override;
  int modifier_solution(DoubleTab&) override;
  void completer(const Equation_base& ) override;
  inline const Equation_base& equation() const;

protected :
  REF(Equation_base) mon_equation;
  REF(Domaine_PolyMAC_P0P1NC) le_dom_PolyMAC_P0P1NC;
  REF(Domaine_Cl_PolyMAC) le_dom_Cl_PolyMAC_P0P1NC;
  int has_P_ref = 0;
  int stencil_done = 0;
  IntVect tab1, tab2;//tableaux tab1 / tab2 de la Matrice_Morse (ne changent pas)
};

inline const Equation_base& Assembleur_P_PolyMAC_P0P1NC::equation() const
{
  return  mon_equation.valeur();
}

#endif
