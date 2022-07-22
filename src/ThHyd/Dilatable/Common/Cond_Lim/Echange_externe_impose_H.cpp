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

#include <Echange_externe_impose_H.h>
#include <Fluide_Dilatable_base.h>
#include <Equation_base.h>
#include <Motcle.h>

Implemente_instanciable(Echange_externe_impose_H,"Paroi_echange_externe_impose_H",Echange_externe_impose);

/*! @brief Ecrit le type de l'objet sur un flot de sortie
 *
 * @param (Sortie& s) un flot de sortie
 * @return (Sortie&) le flot de sortie modifie
 */
Sortie& Echange_externe_impose_H::printOn(Sortie& s ) const
{
  return s << que_suis_je() << "\n";
}

/*! @brief Simple appel a Echange_impose_base::readOn(Entree&) Lit les specifications des conditions aux limites
 *
 *     a partir d'un flot d'entree.
 *
 * @param (Entree& s) un flot d'entree
 * @return (Entree&) le flot de sortie modifie
 */
Entree& Echange_externe_impose_H::readOn(Entree& s )
{
  return Echange_externe_impose::readOn(s) ;
}

/*! @brief Complete les conditions aux limites.
 *
 */
void Echange_externe_impose_H::completer()
{
  Echange_impose_base::completer();
  le_fluide = ref_cast(Fluide_Dilatable_base,ma_zone_cl_dis->equation().milieu());
  modifier_val_imp = 1;
}

/*! @brief Renvoie la valeur de la temperature imposee sur la i-eme composante du champ de frontiere.
 *
 * @param (int i) l'indice de la composante du champ de de frontiere
 * @return (double)
 */
double Echange_externe_impose_H::T_ext(int i) const
{
  if (le_champ_front.valeurs().size()==1)
    {
      if (modifier_val_imp==1)
        return le_fluide->calculer_H(le_champ_front(0,0));
      else
        return le_champ_front(0,0);
    }
  else if (le_champ_front.valeurs().dimension(1)==1)
    {
      if (modifier_val_imp==1)
        return le_fluide->calculer_H(le_champ_front(i,0));
      else
        return le_champ_front(i,0);
    }
  else
    Cerr << "Echange_impose_base::T_ext erreur" << finl;

  abort();
  return 0.;
}

/*! @brief Renvoie la valeur de la temperature imposee sur la (i,j)-eme composante du champ de frontiere.
 *
 * @param (int i)
 * @param (int j)
 * @return (double)
 */
double Echange_externe_impose_H::T_ext(int i, int j) const
{
  if (le_champ_front.valeurs().dimension(0)==1)
    {
      if (modifier_val_imp==1)
        return le_fluide->calculer_H(le_champ_front(0,j));
      else
        return le_champ_front(0,j);
    }
  else
    {
      if (modifier_val_imp==1)
        return le_fluide->calculer_H(le_champ_front(i,j));
      else
        return le_champ_front(i,j);
    }
}

/*! @brief Verifie la compatibilite des conditions aux limites avec l'equation passee en parametre.
 *
 *    Les conditions aux limites de type Ech_imp_base  sont
 *    compatibles avec des equations de type:
 *          - Thermique_H  (thermique avec inconnue = enthalpie)
 *
 * @param (Equation_base& eqn) l'equation avec laquelle on doit verifier la compatibilite
 * @return (int) valeur booleenne, 1 si compatible 0 sinon
 */
int Echange_externe_impose_H::compatible_avec_eqn(const Equation_base& eqn) const
{
  Motcle dom_app=eqn.domaine_application();
  Motcle Thermique="Thermique_H";
  if ( (dom_app==Thermique))
    return 1;
  else
    {
      err_pas_compatible(eqn);
      return 0;
    }
}
