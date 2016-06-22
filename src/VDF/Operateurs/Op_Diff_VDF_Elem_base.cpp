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
// File:        Op_Diff_VDF_Elem_base.cpp
// Directory:   $TRUST_ROOT/src/VDF/Operateurs
// Version:     /main/13
//
//////////////////////////////////////////////////////////////////////////////

#include <Op_Diff_VDF_Elem_base.h>
#include <Champ_P0_VDF.h>

#include <Eval_Diff_VDF.h>
#include <Eval_VDF_Elem.h>

Implemente_base_sans_constructeur(Op_Diff_VDF_Elem_base,"Op_Diff_VDF_Elem_base",Op_Diff_VDF_base);



//// printOn
//

Sortie& Op_Diff_VDF_Elem_base::printOn(Sortie& s ) const
{
  return s << que_suis_je() ;
}


//// readOn
//

Entree& Op_Diff_VDF_Elem_base::readOn(Entree& s )
{
  return s ;
}

double Op_Diff_VDF_Elem_base::calculer_dt_stab() const
{
  // Calcul du pas de temps de stabilite :
  //
  //  - La  diffusivite est uniforme donc :
  //
  //     dt_stab = 1/(2*diffusivite*Max(coef(elem)))
  //
  //     avec:
  //           coef = 1/(dx*dx) + 1/(dy*dy) + 1/(dz*dz)
  //
  //           i decrivant l'ensemble des elements du maillage
  //
  // Rq : On ne balaie pas l'ensemble des elements puisque
  //      le max de coeff est atteint sur l'element qui realise
  //      a la fois le min de dx le min de dy et le min de dz

  const Champ_base& diff_dt=diffusivite_pour_pas_de_temps();

  double dt_stab = DMAXFLOAT;
  double coef;
  const Zone_VDF& zone_VDF = iter.zone();
  if (sub_type(Champ_Uniforme,diff_dt))
    {
      // GF le max permet de traiter le multi_inco
      double alpha=max_array(diff_dt.valeurs());

      coef = 1/(zone_VDF.h_x()*zone_VDF.h_x())
             + 1/(zone_VDF.h_y()*zone_VDF.h_y());

      if (dimension == 3)
        coef += 1/(zone_VDF.h_z()*zone_VDF.h_z());
      if (alpha==0)
        dt_stab = DMAXFLOAT;
      else
        dt_stab = 0.5/(alpha*coef);

    }
  else
    {
      const DoubleTab& alpha = diff_dt.valeurs();
      double dt_loc;
      double h, l;
      double alpha_loc;
      int nbdim=alpha.nb_dim();
      int i, elem, n=alpha.dimension(0);
      for (elem=0 ; elem<n ; elem++)
        {
          h = 0;
          for (i=0 ; i<dimension ; i++)
            {
              l = zone_VDF.dim_elem(elem,i);
              h += 1./(l*l);
            }
          if (nbdim==1)
            alpha_loc=alpha(elem);
          else
            {
              alpha_loc = alpha(elem,0);
              for (int ncomp=1; ncomp<alpha.dimension(1); ncomp++)
                alpha_loc = max(alpha_loc,alpha(elem,ncomp));

            }
          dt_loc = 0.5/((alpha_loc+DMINFLOAT)*h);
          if (dt_loc<dt_stab)
            dt_stab = dt_loc;
        }
    }
  return Process::mp_min(dt_stab);

}


// Description:
// complete l'iterateur et l'evaluateur
void Op_Diff_VDF_Elem_base::associer(const Zone_dis& zone_dis,
                                     const Zone_Cl_dis& zone_cl_dis,
                                     const Champ_Inc& ch_diffuse)
{
  const Champ_P0_VDF& inco = ref_cast(Champ_P0_VDF,ch_diffuse.valeur());
  const Zone_VDF& zvdf = ref_cast(Zone_VDF,zone_dis.valeur());
  const Zone_Cl_VDF& zclvdf = ref_cast(Zone_Cl_VDF,zone_cl_dis.valeur());
  iter.associer(zvdf, zclvdf, *this);

  Evaluateur_VDF& iter_vdf=iter.evaluateur();
  iter_vdf.associer_zones(zvdf, zclvdf );          // Evaluateur_VDF::associer_zones
  // ((Eval_VDF_Elem&) iter.evaluateur()).associer_inconnue(inco );        // Eval_VDF_Elem::associer_inconnue
  // GF sans dynamic_cast il FAUT rajouter une methode associer_inconnue
  // la ligne du dessu fait core_dump (j'ai eu le meme pb avec ocnd_lim_rayo)
  //(dynamic_cast<Eval_VDF_Elem&> (iter_vdf)).associer_inconnue(inco );
  get_eval_elem().associer_inconnue(inco );
}


// Description:
// associe le champ de diffusivite a l'evaluateur
void Op_Diff_VDF_Elem_base::associer_diffusivite(const Champ_base& ch_diff)
{
  Eval_Diff_VDF& eval_diff = (Eval_Diff_VDF&) iter.evaluateur();
  ////Eval_Diff_VDF& eval_diff = (Eval_Diff_VDF&) dynamic_cast<Eval_Diff_VDF&>(iter.evaluateur());

  eval_diff.associer(ch_diff);                // Eval_Diff_VDF::associer
}

const Champ_base& Op_Diff_VDF_Elem_base::diffusivite() const
{
  const Eval_Diff_VDF& eval_diff = (const Eval_Diff_VDF&) iter.evaluateur();
  ////const Eval_Diff_VDF& eval_diff = (const Eval_Diff_VDF&) dynamic_cast<const Eval_Diff_VDF&>(iter.evaluateur());
  return eval_diff.get_diffusivite();
}
