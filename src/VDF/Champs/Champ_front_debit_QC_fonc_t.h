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

#ifndef Champ_front_debit_QC_fonc_t_included
#define Champ_front_debit_QC_fonc_t_included

#include <Ch_front_var_instationnaire_indep.h>
#include <TRUST_Vector.h>
#include <TRUST_Ref.h>
#include <Parser_U.h>

class Fluide_Quasi_Compressible;

/*! @brief classe Champ_front_debit_QC_fonc_t Classe derivee de Champ_front_base qui represente les
 *
 *      champs aux frontieres debit_QCs : ch(*,i)=m(i)+alea*amplitude(i)
 *      ou alea est une variable aleatoire de loi uniforme sur [0,1]
 *
 * @sa Champ_front_base
 */
class Champ_front_debit_QC_fonc_t : public Ch_front_var_instationnaire_indep
{
  Declare_instanciable(Champ_front_debit_QC_fonc_t);

public:

  Champ_front_base& affecter_(const Champ_front_base& ch) override;
  void mettre_a_jour(double temps) override;
  virtual double valeur_au_temps(double temps, int som, int comp) const;

protected :
  int ismoyen=0;
  OBS_PTR(Fluide_Quasi_Compressible) fluide;

private :
  mutable VECT(Parser_U) f_debit_t;
};

#endif

