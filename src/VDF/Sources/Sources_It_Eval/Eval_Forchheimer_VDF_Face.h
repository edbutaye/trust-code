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

#ifndef Eval_Forchheimer_VDF_Face_included
#define Eval_Forchheimer_VDF_Face_included

#include <Modele_Permeabilite_base.h>
#include <Evaluateur_Source_Face.h>
#include <Champ_Inc_base.h>
#include <TRUST_Ref.h>

class Eval_Forchheimer_VDF_Face: public Evaluateur_Source_Face
{
public:
  Eval_Forchheimer_VDF_Face() : Cf_(1.), porosite_(1.) { }
  Eval_Forchheimer_VDF_Face(const Eval_Forchheimer_VDF_Face& eval) : Evaluateur_Source_Face(eval), Cf_(1.), porosite_(1.) { }
  template <typename Type_Double> void calculer_terme_source(const int , Type_Double& ) const;
  template <typename Type_Double> void calculer_terme_source_bord(const int num_face, Type_Double& source) const { calculer_terme_source(num_face,source); }
  inline void mettre_a_jour() override { /* Do nothing */}
  inline void setCf(double c) { Cf_ = c; }
  inline void associer(const Champ_Inc_base& vit) { vitesse_ = vit;}
  inline void setPorosite(double p) { porosite_ = p; }

  OWN_PTR(Modele_Permeabilite_base) modK_;

protected:
  OBS_PTR(Champ_Inc_base) vitesse_;
  double Cf_, porosite_;
};

template <typename Type_Double>
void Eval_Forchheimer_VDF_Face::calculer_terme_source(const int num_face, Type_Double& source) const
{
  const int size = source.size_array();
  for (int i = 0; i < size; i++)
    {
      const double U = (vitesse_->valeurs())(num_face,i);
      source[i] = -Cf_/sqrt(modK_->getK(porosite_))*volumes_entrelaces(num_face)*porosite_surf(num_face)*std::fabs(U)*U;
    }
}

#endif /* Eval_Forchheimer_VDF_Face_included */
