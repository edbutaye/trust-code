/****************************************************************************
* Copyright (c) 2024, CEA
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

#include <Neumann_sortie_libre_Temp_H.h>
#include <Fluide_Dilatable_base.h>
#include <Equation_base.h>

Implemente_instanciable(Neumann_sortie_libre_Temp_H, "Sortie_libre_temperature_imposee_H", Neumann_sortie_libre);
// XD sortie_libre_temperature_imposee_h neumann sortie_libre_temperature_imposee_h 0 Open boundary for heat equation with enthalpy as unknown.
// XD attr ch front_field_base ch 0 Boundary field type.


Sortie& Neumann_sortie_libre_Temp_H::printOn(Sortie& s) const { return s << que_suis_je() << finl; }

Entree& Neumann_sortie_libre_Temp_H::readOn(Entree& s)
{
  if (app_domains.size() == 0) app_domains = { Motcle("Thermique_H") };

  s >> le_champ_ext;
  le_champ_front = le_champ_ext;

  return s;
}

/*! @brief Complete les conditions aux limites.
 *
 */
void Neumann_sortie_libre_Temp_H::completer()
{
  le_fluide = ref_cast(Fluide_Dilatable_base, mon_dom_cl_dis->equation().milieu());
  modifier_val_imp = 1;
}

/*! @brief Renvoie la valeur de la i-eme composante du champ impose a l'exterieur de la frontiere.
 *
 * @param (int i) indice suivant la premiere dimension du champ
 * @return (double) la valeur imposee sur la composante du champ specifiee
 * @throws deuxieme dimension du champ de frontiere superieur a 1
 */
double Neumann_sortie_libre_Temp_H::val_ext(int i) const
{
  if (le_champ_ext->valeurs().size() == 1)
    {
      if (modifier_val_imp == 1)
        return le_fluide->calculer_H(le_champ_ext->valeurs()(0, 0));
      else
        return le_champ_ext->valeurs()(0, 0);
    }
  else if (le_champ_ext->valeurs().dimension(1) == 1)
    {
      if (modifier_val_imp == 1)
        return le_fluide->calculer_H(le_champ_ext->valeurs()(i, 0));
      else
        return le_champ_ext->valeurs()(i, 0);
    }
  else
    {
      Cerr << "Neumann_sortie_libre_Temp_H::val_ext" << finl;
      Cerr << le_champ_ext << finl;
    }

  abort();
  return 0.;
}

/*! @brief Renvoie la valeur de la (i,j)-eme composante du champ impose a l'exterieur de la frontiere.
 *
 * @param (int i) indice suivant la premiere dimension du champ
 * @param (int j) indice suivant la deuxieme dimension du champ
 * @return (double) la valeur imposee sur la composante du champ specifiee
 */
double Neumann_sortie_libre_Temp_H::val_ext(int i, int j) const
{
  if (le_champ_ext->valeurs().dimension(0) == 1)
    {
      if (modifier_val_imp == 1)
        return le_fluide->calculer_H(le_champ_ext->valeurs()(0, j));
      else
        return le_champ_ext->valeurs()(0, j);
    }
  else
    {
      if (modifier_val_imp == 1)
        return le_fluide->calculer_H(le_champ_ext->valeurs()(i, j));
      else
        return le_champ_ext->valeurs()(i, j);
    }
}
