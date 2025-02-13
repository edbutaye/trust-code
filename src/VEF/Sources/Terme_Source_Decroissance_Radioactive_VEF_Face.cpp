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

#include <Terme_Source_Decroissance_Radioactive_VEF_Face.h>
#include <Equation_base.h>

#include <Domaine_VEF.h>
#include <Synonyme_info.h>

Implemente_instanciable_sans_constructeur(Terme_Source_Decroissance_Radioactive_VEF_Face,"Decroissance_Radioactive_VEF_P1NC",Source_base);
Add_synonym(Terme_Source_Decroissance_Radioactive_VEF_Face,"Radioactive_Decay_VEF_P1NC");


Sortie& Terme_Source_Decroissance_Radioactive_VEF_Face::printOn(Sortie& s ) const
{
  return s << que_suis_je();
}

Entree& Terme_Source_Decroissance_Radioactive_VEF_Face::readOn(Entree& s)
{
  double lambda_tmp;
  int nb_groupes;
  s >> nb_groupes;
  Cerr << "Nombre de groupes a lire : " << nb_groupes << finl;
  for (int i = 0; i < nb_groupes; i++)
    {
      s >> lambda_tmp;
      Cerr << "lambda lu : " << lambda_tmp << finl;
      lambda.push_back(lambda_tmp);
    }

  const int N = equation().inconnue().valeurs().line_size(), ng = (int)lambda.size();
  if (N != ng)
    {
      Cerr << "Terme_Source_Decroissance_Radioactive_Elem_PolyMAC : inconsistency between the number of radioactive decay constants ( " << ng
           << " ) and the number of components of the unknown of the equation ( " << N << " )" << finl;
      Process::exit();
    }
  return s ;
}

void Terme_Source_Decroissance_Radioactive_VEF_Face::associer_domaines(const Domaine_dis_base& domaine_dis,
                                                                       const Domaine_Cl_dis_base& domaine_Cl_dis)
{
  Cerr << " Terme_Source_Decroissance_Radioactive_VEF_Face::associer_domaines " << finl ;
  le_dom_VEF = ref_cast(Domaine_VEF, domaine_dis);
}

DoubleTab& Terme_Source_Decroissance_Radioactive_VEF_Face::ajouter(DoubleTab& resu)  const
{
  const Domaine_VF& domaine = le_dom_VEF.valeur();
  const DoubleVect& vf = domaine.volumes_entrelaces();
  const DoubleTab& c = equation().inconnue().valeurs();
  const int nb_faces = le_dom_VEF->nb_faces(), N = c.line_size();

  for (int f = 0; f < nb_faces; f++)
    for (int l = 0; l < N; l++)
      resu(f, l) -= lambda[l] * c(f, l) * vf(f);

  return resu;
}

DoubleTab& Terme_Source_Decroissance_Radioactive_VEF_Face::calculer(DoubleTab& resu) const
{
  resu=0;
  ajouter(resu);
  return resu;
}

void Terme_Source_Decroissance_Radioactive_VEF_Face::contribuer_a_avec(const DoubleTab& inco, Matrice_Morse& matrice) const
{
  const Domaine_VF& domaine = le_dom_VEF.valeur();
  const DoubleVect& vf = domaine.volumes_entrelaces();
  const int nb_faces = le_dom_VEF->nb_faces(), N = equation().inconnue().valeurs().line_size();

  for (int f = 0; f < nb_faces; f++)
    for (int l = 0; l < N; l++)
      {
        const int k = f * N + l;
        matrice(k, k) += lambda[l] * vf(f);
      }
}
