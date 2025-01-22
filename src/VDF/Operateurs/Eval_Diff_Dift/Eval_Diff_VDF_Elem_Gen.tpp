/****************************************************************************
* Copyright (c) 2025, CEA
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

#ifndef Eval_Diff_VDF_Elem_Gen_TPP_included
#define Eval_Diff_VDF_Elem_Gen_TPP_included

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::flux_face(const DoubleTab& inco, const DoubleTab& val_b, const int face, const Dirichlet_entree_fluide& la_cl, const int num1, Type_Double& flux) const
{
  // Olga avait mis : double dist = 2*Dist_norm_bord(face);
  // Pierre dit que :
  const double dist = Dist_norm_bord(face);
  const int i = elem_(face,0), j = elem_(face,1), ncomp = flux.size_array();

  for (int k = 0; k < ncomp; k++)
    {
      if (DERIVED_T::IS_QUASI)
        {
          const double T_imp = la_cl.val_imp(face - num1, k);
          const int ori = DERIVED_T::IS_ANISO ? orientation(face) : k;
          flux[k] = (i != -1) ? (T_imp - inco(i, k)) / dv_mvol(i) * surface(face) * porosite(face) * nu_1(i, ori) / dist :
                    (inco(j, k) - T_imp) / dv_mvol(j) * surface(face) * porosite(face) * nu_1(j, ori) / dist;
        }
      else if (DERIVED_T::IS_MULTI_SCALAR_DIFF)
        {
          flux[k] = 0.0;
          for (int l = 0; l < ncomp; l++)
            {
              const double T_imp = la_cl.val_imp(face - num1, l);
              const int ori = ncomp * k + l;

              flux[k] += (i != -1) ? (T_imp - inco(i, l)) * surface(face) * porosite(face) * nu_1(i, ori) / dist :
                         (inco(j, l) - T_imp) * surface(face) * porosite(face) * nu_1(j, ori) / dist;
            }
        }
      else
        {
          const double T_imp = la_cl.val_imp(face - num1, k);
          const int ori = DERIVED_T::IS_ANISO ? orientation(face) : k;

          flux[k] = (i != -1) ? (T_imp - inco(i, k)) * surface(face) * porosite(face) * nu_1(i, ori) / dist :
                    (inco(j, k) - T_imp) * surface(face) * porosite(face) * nu_1(j, ori) / dist;
        }
    }
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::flux_face(const DoubleTab& inco, const DoubleTab& val_b, const int face, const Scalaire_impose_paroi& la_cl, const int num1, Type_Double& flux) const
{
  const double dist = Dist_norm_bord(face);
  const int i = elem_(face,0), j = elem_(face,1), ncomp = flux.size_array();

  for (int k = 0; k < ncomp; k++)
    {
      if (DERIVED_T::IS_MULTI_SCALAR_DIFF)
        {
          flux[k] = 0.0;
          for (int l = 0; l < ncomp; l++)
            {
              const double T_imp = la_cl.val_imp(face-num1, l);
              const int ori = ncomp * k + l;
              flux[k] += (i != -1) ? (T_imp-inco(i,l))*surface(face)*porosite(face)*nu_1(i,ori)/dist :
                         (inco(j,l)-T_imp)*surface(face)*porosite(face)*nu_1(j,ori)/dist;
            }
        }
      else
        {
          const double T_imp = la_cl.val_imp(face-num1,k);
          const int ori = DERIVED_T::IS_ANISO ? orientation(face) : k;
          flux[k] = (i != -1) ? (T_imp-inco(i,k))*surface(face)*porosite(face)*nu_1(i,ori)/dist :
                    (inco(j,k)-T_imp)*surface(face)*porosite(face)*nu_1(j,ori)/dist;
        }
    }
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::flux_face(const DoubleTab& inco, const DoubleTab& val_b, const int face, const Dirichlet_loi_paroi& la_cl, const int num1, Type_Double& flux) const
{
  if (DERIVED_T::IS_MULTI_SCALAR_DIFF) throw;

  const double dist = Dist_norm_bord(face);
  const int i = elem_(face,0), j = elem_(face,1), ncomp = flux.size_array();

  for (int k = 0; k < ncomp; k++)
    {
      const double T_imp = la_cl.val_imp(face-num1,k);
      const int ori = DERIVED_T::IS_ANISO ? orientation(face) : k;
      flux[k] = (i != -1) ? (T_imp-inco(i,k))*surface(face)*porosite(face)*nu_1(i,ori)/dist : (inco(j,k)-T_imp)*surface(face)*porosite(face)*nu_1(j,ori)/dist;
    }
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::flux_face(const DoubleTab& , const DoubleTab& val_b, const int face, const Neumann_paroi& la_cl, const int num1, Type_Double& flux) const
{
  if (DERIVED_T::IS_MULTI_SCALAR_DIFF) throw;

  const int i = elem_(face,0), ncomp = flux.size_array();
  for (int k=0; k < ncomp; k++) flux[k] = ((i != -1) ? 1 : -1) * la_cl.flux_impose(face-num1,k)*surface(face);
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::flux_face(const DoubleTab& inco, const DoubleTab&, const int face, const Periodique& la_cl, const int , Type_Double& flux) const
{
  if (DERIVED_T::IS_MULTI_SCALAR_DIFF) throw;

  const int i = elem_(face,0), j = elem_(face,1), ncomp = flux.size_array();
  const double d0 = le_dom->dist_face_elem0_period(face,i,la_cl.distance()), d1 = le_dom->dist_face_elem1_period(face,j,la_cl.distance());

  for (int k = 0; k < ncomp; k++)
    {
      const int ori = DERIVED_T::IS_ANISO ? orientation(face) : k;
      const double heq = compute_heq(d0,i,d1,j,ori);

      flux[k] = DERIVED_T::IS_QUASI ? heq*(inco(j,k)/dv_mvol(j) - inco(i,k)/dv_mvol(i))*surface(face)*porosite(face) : heq*(inco(j,k) - inco(i,k))*surface(face)*porosite(face);
    }
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::flux_face(const DoubleTab& inco, const DoubleTab& val_b, const int face, const Dirichlet_paroi_fixe&, const int num1, Type_Double& flux ) const
{
  if (DERIVED_T::IS_MULTI_SCALAR_DIFF) throw;

  const int i = elem_(face,0), j = elem_(face,1), ncomp = flux.size_array();
  const double dist = Dist_norm_bord(face);

  if (DERIVED_T::IS_QUASI)
    for (int k = 0; k < ncomp; k++) flux[k] = (i != -1) ? -inco(i,k)*surface(face)*porosite(face)*nu_1(i,k)/dv_mvol(i)/dist : inco(j,k)*surface(face)*porosite(face)*nu_1(j,k)/dv_mvol(j)/dist;
  else
    for (int k = 0; k < ncomp; k++) flux[k] = (i != -1) ? -inco(i,k)*surface(face)*porosite(face)*nu_1(i,k)/dist : inco(j,k)*surface(face)*porosite(face)*nu_1(j,k)/dist;
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::flux_face(const DoubleTab& inco, const DoubleTab&, const int face , const Echange_global_impose& la_cl, const int num1, Type_Double& flux) const
{
  if (DERIVED_T::IS_MULTI_SCALAR_DIFF) throw;
  if (DERIVED_T::IS_QUASI)  not_implemented_k_eps(__func__);

  const int i = elem_(face,0), j = elem_(face,1), ncomp = flux.size_array();

  for (int k = 0; k < ncomp; k++)
    {
      const double h = la_cl.h_imp(face-num1,k), T_ext = la_cl.T_ext(face-num1,k);
      const double phi_ext = la_cl.has_phi_ext() ? la_cl.flux_exterieur_impose(face-num1,k) : 0;
      flux[k] = (i != -1) ? (phi_ext+h*(T_ext-inco(i,k)))*surface(face) : (-phi_ext+h*(inco(j,k)-T_ext))*surface(face);
    }
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::flux_face(const DoubleTab& inco, const int boundary_index, const int face, const int local_face, const Echange_externe_impose& la_cl, const int num1, Type_Double& flux) const
{
  if (DERIVED_T::IS_QUASI) not_implemented_k_eps(__func__);

  // C.L de type Echange_externe_impose : 1/h_total = (1/h_imp) + (e/diffusivite) : La C.L fournit h_imp ; il faut calculer e/diffusivite
  double heq, h_total_inv, e;
  const int i = elem_(face,0), j = elem_(face,1), ncomp = flux.size_array();

  if (DERIVED_T::IS_MODIF_DEQ) e = ind_Fluctu_Term()==1 ? Dist_norm_bord_externe_(face) : equivalent_distance(boundary_index,local_face);
  else e = DERIVED_T::IS_DEQUIV ? equivalent_distance(boundary_index,local_face) : Dist_norm_bord_externe_(face);

  int elem_opp = -1;
  if (sub_type(Echange_interne_impose, la_cl))
    {
      // In this case, T_ext might change within the timestep (e.g. Newton's algorithm used for the
      // variable gap model). Make sure we get the most up-to-date value.
      const Echange_interne_impose& cl = ref_cast(Echange_interne_impose, la_cl);
      const Champ_front_calc_interne& Text = ref_cast(Champ_front_calc_interne, cl.T_ext());
      const IntVect& fmap = Text.face_map();
      int opp_face = fmap[face-num1]+num1;  // num1 is the index of the first face
      int elem1 = le_dom->face_voisins(opp_face, 0);
      elem_opp = (elem1 != -1) ? elem1 : le_dom->face_voisins(opp_face, 1);
    }

  // XXX : E. Saikali 08/03/2021 : The test of a zero diffusion was not done before. I think it should be like that
  // Attention for DIFT OPERATORS : nu_2 and not nu_1 (only laminar part)
  if ( i != -1 )
    {
      for (int k = 0; k < ncomp; k++)
        {
          flux[k] = 0.0;
          for (int l = (DERIVED_T::IS_MULTI_SCALAR_DIFF ? 0 : k); l < (DERIVED_T::IS_MULTI_SCALAR_DIFF ? ncomp : k + 1); l++)
            {
              const int ori = DERIVED_T::IS_MULTI_SCALAR_DIFF ? (ncomp * k + l) : (DERIVED_T::IS_ANISO ? orientation(face) : k);
              const double h_imp = la_cl.h_imp(face - num1, DERIVED_T::IS_MULTI_SCALAR_DIFF ? ori : k), T_ext = (elem_opp == -1) ? la_cl.T_ext(face-num1, l) : inco(elem_opp, l);
              if ( nu_2(i, ori) == 0.0 ) heq = 0.0;
              else
                {
                  h_total_inv =  1.0/h_imp + e/nu_2(i, ori);
                  heq = 1.0 / h_total_inv;
                }
              flux[k] += heq * (T_ext - inco(i, l)) * surface(face);
            }
        }
    }
  else // j != -1
    {
      for (int k = 0; k < ncomp; k++)
        {
          flux[k] = 0.0;
          for (int l = (DERIVED_T::IS_MULTI_SCALAR_DIFF ? 0 : k); l < (DERIVED_T::IS_MULTI_SCALAR_DIFF ? ncomp : k + 1); l++)
            {
              const int ori = DERIVED_T::IS_MULTI_SCALAR_DIFF ? (ncomp * k + l) : (DERIVED_T::IS_ANISO ? orientation(face) : k);
              const double h_imp = la_cl.h_imp(face - num1, DERIVED_T::IS_MULTI_SCALAR_DIFF ? ori : k), T_ext = (elem_opp == -1) ? la_cl.T_ext(face-num1, l) : inco(elem_opp, l);
              if ( nu_2(j,ori) == 0.0 ) heq = 0.0;
              else
                {
                  h_total_inv = 1.0/h_imp + e/nu_2(j,ori);
                  heq = 1.0 / h_total_inv;
                }
              flux[k] += heq*(inco(j,l)-T_ext)*surface(face);
            }
        }
    }
}

inline double newton_Tbord(double eps, double T_ext, double lambda, double e, double T0, double sgn)
{
  double Tb = T0;
  const double it_max = 100, tolerance = 1e-5;
  for (int i = 0; i < it_max; i++)
    {
      const double f = sgn * (5.67e-8 * eps * (T_ext * T_ext * T_ext * T_ext - Tb * Tb * Tb * Tb) - lambda / e * (Tb - T0));
      const double f_p = sgn * (-4 * 5.67e-8 * eps * Tb * Tb * Tb - lambda / e);
      const double Tb_new = Tb - f / f_p;
      if (std::abs(Tb_new - Tb) < tolerance)
        return Tb_new;
      Tb = Tb_new;
    }
  Process::exit("Echange_Externe_radiatif : newton did not converge !");
  return Tb;
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::flux_face(const DoubleTab& inco, const int boundary_index, const int face, const int local_face, const Echange_externe_radiatif& la_cl, const int num1, Type_Double& flux) const
{
  if (DERIVED_T::IS_QUASI) not_implemented_k_eps(__func__);

  double e;
  const int i = elem_(face,0), j = elem_(face,1), ncomp = flux.size_array();

  if (DERIVED_T::IS_MODIF_DEQ) e = ind_Fluctu_Term()==1 ? Dist_norm_bord_externe_(face) : equivalent_distance(boundary_index,local_face);
  else e = DERIVED_T::IS_DEQUIV ? equivalent_distance(boundary_index,local_face) : Dist_norm_bord_externe_(face);


  // XXX : E. Saikali 08/03/2021 : The test of a zero diffusion was not done before. I think it should be like that
  // Attention for DIFT OPERATORS : nu_2 and not nu_1 (only laminar part)
  if (i != -1)
    {
      for (int k = 0; k < ncomp; k++)
        {
          flux[k] = 0.0;
          for (int l = (DERIVED_T::IS_MULTI_SCALAR_DIFF ? 0 : k); l < (DERIVED_T::IS_MULTI_SCALAR_DIFF ? ncomp : k + 1); l++)
            {
              const int ori = DERIVED_T::IS_MULTI_SCALAR_DIFF ? (ncomp * k + l) : (DERIVED_T::IS_ANISO ? orientation(face) : k);
              const double eps = la_cl.emissivite(face - num1, DERIVED_T::IS_MULTI_SCALAR_DIFF ? ori : k), T_ext = la_cl.T_ext(face - num1, l);
              const double Tb = newton_Tbord(eps, T_ext, nu_2(i, ori), e, inco(i, l), 1.0);
              flux[k] += 5.67e-8 * eps * (T_ext * T_ext * T_ext * T_ext - Tb * Tb * Tb * Tb) * surface(face);
            }
        }
    }
  else // j != -1
    {
      for (int k = 0; k < ncomp; k++)
        {
          flux[k] = 0.0;
          for (int l = (DERIVED_T::IS_MULTI_SCALAR_DIFF ? 0 : k); l < (DERIVED_T::IS_MULTI_SCALAR_DIFF ? ncomp : k + 1); l++)
            {
              const int ori = DERIVED_T::IS_MULTI_SCALAR_DIFF ? (ncomp * k + l) : (DERIVED_T::IS_ANISO ? orientation(face) : k);
              const double eps = la_cl.emissivite(face - num1, DERIVED_T::IS_MULTI_SCALAR_DIFF ? ori : k), T_ext = la_cl.T_ext(face - num1, l);
              const double Tb = newton_Tbord(eps, T_ext, nu_2(i, ori), e, inco(j, l), -1.0);
              flux[k] += 5.67e-8 * eps * (Tb * Tb * Tb * Tb - T_ext * T_ext * T_ext * T_ext) * surface(face);
            }
        }
    }
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::flux_faces_interne(const DoubleTab& inco, const int face, Type_Double& flux) const
{
  const int i = elem_(face,0), j = elem_(face,1), ncomp = flux.size_array();
  double heq, d0 = Dist_face_elem0(face,i), d1 = Dist_face_elem1(face,j);
  for (int k = 0; k < ncomp; k++)
    {
      const int ori = DERIVED_T::IS_ANISO ? orientation(face) : k;
      if (DERIVED_T::IS_RANS)
        {
          heq = compute_heq(d0,i, d1,j,ori); // pas d'assert pour k-eps !
          flux[k] = DERIVED_T::IS_QUASI ? heq*(inco(j,k)/dv_mvol(j) - inco(i,k)/dv_mvol(i))*surface(face)*porosite(face) :
                    heq*(inco(j,k)-inco(i,k))*surface(face)*porosite(face);
        }
      else if (DERIVED_T::IS_MULTI_SCALAR_DIFF)
        {
          flux[k] = 0.0;
          for (int l = 0; l < ncomp; l++)
            {
              heq = compute_heq(d0, i, d1, j, ncomp * k + l);
              flux[k] += heq * (inco(j, l) - inco(i, l)) * surface(face) * porosite(face);
            }
        }
      else
        {
          if (nu_1(i,ori) == 0.0 && nu_1(j,ori) == 0.0) heq = 0.;
          else
            {
              assert(nu_1(i,ori) != 0.0 && nu_1(j,ori) != 0.0);
              heq = compute_heq(d0,i, d1,j,ori);
            }
          flux[k] = heq*(inco(j,k)-inco(i,k))*surface(face)*porosite(face);
        }
    }
}

/* ************************************** *
 * *********  POUR L'IMPLICITE ********** *
 * ************************************** */

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::coeffs_face(const int face, const int, const Dirichlet_entree_fluide& la_cl, Type_Double& aii, Type_Double& ajj) const
{
  assert (aii.size_array() == ajj.size_array());
  const int i = elem_(face,0), j = elem_(face,1),
            ncomp = DERIVED_T::IS_MULTI_SCALAR_DIFF ? static_cast<int>(std::sqrt(aii.size_array())) : aii.size_array();

  const double dist = Dist_norm_bord(face);

  for (int k = 0; k < ncomp; k++)
    if (DERIVED_T::IS_MULTI_SCALAR_DIFF)
      {
        for (int l = 0; l < ncomp; l++)
          {
            const int ori = ncomp * k + l;
            aii[ori] = (i != -1) ? porosite(face) * nu_1(i, ori) * surface(face) / dist : 0.;
            ajj[ori] = (i != -1) ? 0. : porosite(face) * nu_1(j, ori) * surface(face) / dist;
          }
      }
    else
      {
        const int ori = DERIVED_T::IS_ANISO ? orientation(face) : k;
        aii[k] = (i != -1) ? porosite(face) * nu_1(i, ori) * surface(face) / dist : 0.;
        ajj[k] = (i != -1) ? 0. : porosite(face) * nu_1(j, ori) * surface(face) / dist;
      }
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::coeffs_face(const int face, const int, const Scalaire_impose_paroi& la_cl, Type_Double& aii, Type_Double& ajj) const
{
  assert (aii.size_array() == ajj.size_array());
  const int i = elem_(face,0), j = elem_(face,1),
            ncomp = DERIVED_T::IS_MULTI_SCALAR_DIFF ? static_cast<int>(std::sqrt(aii.size_array())) : aii.size_array();

  const double dist = Dist_norm_bord(face);

  for (int k = 0; k < ncomp; k++)
    if (DERIVED_T::IS_MULTI_SCALAR_DIFF)
      {
        for (int l = 0; l < ncomp; l++)
          {
            const int ori = ncomp * k + l;
            aii[ori] = (i != -1) ? porosite(face) * nu_1(i, ori) * surface(face) / dist : 0.;
            ajj[ori] = (i != -1) ? 0. : porosite(face) * nu_1(j, ori) * surface(face) / dist;
          }
      }
    else
      {
        const int ori = DERIVED_T::IS_ANISO ? orientation(face) : k;
        aii[k] = (i != -1) ? porosite(face) * nu_1(i, ori) * surface(face) / dist : 0.;
        ajj[k] = (i != -1) ? 0. : porosite(face) * nu_1(j, ori) * surface(face) / dist;
      }
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::coeffs_face(const int face, const int, const Dirichlet_loi_paroi& la_cl, Type_Double& aii, Type_Double& ajj) const
{
  assert (aii.size_array() == ajj.size_array());
  const int i = elem_(face,0), j = elem_(face,1), ncomp = aii.size_array();
  const double dist = Dist_norm_bord(face);
  for (int k = 0; k < ncomp; k++)
    {
      const int ori = DERIVED_T::IS_ANISO ? orientation(face) : k;
      aii[k] = (i != -1) ? porosite(face)*nu_1(i,ori)*surface(face) / dist : 0.;
      ajj[k] = (i != -1) ? 0. : porosite(face)*nu_1(j,ori)*surface(face) / dist;
    }
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::coeffs_face(const int face, const int, const Periodique& la_cl, Type_Double& aii, Type_Double& ajj ) const
{
  assert (aii.size_array() == ajj.size_array());
  const int i = elem_(face,0), j = elem_(face,1), ncomp = aii.size_array();
  const double d0 = le_dom->dist_face_elem0_period(face,i,la_cl.distance()), d1 = le_dom->dist_face_elem1_period(face,j,la_cl.distance());
  for (int k = 0; k < ncomp; k++)
    {
      const int ori = DERIVED_T::IS_ANISO ? orientation(face) : k;
      const double heq = compute_heq(d0,i,d1,j,ori);
      aii[k] = ajj[k] = heq*surface(face)*porosite(face); // On peut faire ca !
    }
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::coeffs_face(const int face, const int num1, const Dirichlet_paroi_fixe&, Type_Double& aii , Type_Double& ajj) const
{
  assert (aii.size_array() == ajj.size_array());
  const int i = elem_(face,0), j = elem_(face,1), ncomp = aii.size_array();
  const double dist = Dist_norm_bord(face);
  if (DERIVED_T::IS_QUASI)
    {
      for (int k = 0; k < ncomp; k++)
        {
          aii[k] = (i != -1) ? surface(face)*porosite(face)*nu_1(i,k)/dv_mvol(i)/dist : 0.;
          ajj[k] = (i != -1) ? 0. : surface(face)*porosite(face)*nu_1(j,k)/dv_mvol(j)/dist;
        }
    }
  else
    for (int k = 0; k < ncomp; k++)
      {
        aii[k] = (i != -1) ? surface(face)*porosite(face)*nu_1(i,k)/dist : 0.;
        ajj[k] = (i != -1) ? 0. : surface(face)*porosite(face)*nu_1(j,k)/dist;
      }
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::coeffs_face(const int face, const int num1, const Echange_global_impose& la_cl, Type_Double& aii, Type_Double& ajj ) const
{
  assert (aii.size_array() == ajj.size_array());
  const int i = elem_(face,0), ncomp = aii.size_array();
  for (int k = 0; k < ncomp; k++)
    {
      const double h = la_cl.h_imp(face-num1,k);
      const double dphi_dT = la_cl.has_phi_ext() ? la_cl.derivee_flux_exterieur_imposee(face-num1,k) : 0;
      aii[k] = (i != -1) ? (dphi_dT + h)*surface(face) : 0.;
      ajj[k] = (i != -1) ? 0. : (dphi_dT + h)*surface(face);
    }
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::coeffs_face(const int boundary_index, const int face, const int local_face, const int num1, const Echange_externe_impose& la_cl, Type_Double& aii, Type_Double& ajj) const
{
  // C.L de type Echange_externe_impose : 1/h_total = (1/h_imp) + (e/diffusivite) : La C.L fournit h_imp ; il faut calculer e/diffusivite
  assert (aii.size_array() == ajj.size_array());
  const int i = elem_(face,0), j = elem_(face,1), ncomp = DERIVED_T::IS_MULTI_SCALAR_DIFF ? int(sqrt(aii.size_array())) : aii.size_array();
  double e, heq, h_total_inv;

  if (DERIVED_T::IS_MODIF_DEQ) e = ind_Fluctu_Term() == 1 ? Dist_norm_bord_externe_(face) : equivalent_distance(boundary_index,local_face);
  else e = DERIVED_T::IS_DEQUIV ? equivalent_distance(boundary_index,local_face) : Dist_norm_bord_externe_(face);

  const bool is_internal = sub_type(Echange_interne_impose, la_cl) ? true : false;

  if (i != -1)
    for (int k = 0; k < ncomp; k++)
      for (int l = (DERIVED_T::IS_MULTI_SCALAR_DIFF ? 0 : k); l < (DERIVED_T::IS_MULTI_SCALAR_DIFF ? ncomp : k + 1); l++)
        {
          const int ori = DERIVED_T::IS_MULTI_SCALAR_DIFF ? (ncomp * k + l) : (DERIVED_T::IS_ANISO ? orientation(face) : k);
          const double h_imp = la_cl.h_imp(face-num1, DERIVED_T::IS_MULTI_SCALAR_DIFF ? ori : k);
          if (nu_2(i,ori) == 0.0) heq = 0.0;
          else
            {
              h_total_inv =  1.0/h_imp + e/nu_2(i,ori);
              heq = 1.0 / h_total_inv;
            }
          aii[DERIVED_T::IS_MULTI_SCALAR_DIFF ? ncomp * k + l : k] = heq*surface(face);
          ajj[DERIVED_T::IS_MULTI_SCALAR_DIFF ? ncomp * k + l : k] = is_internal ? heq*surface(face) : 0.;
        }
  else
    for (int k = 0; k < ncomp; k++)
      for (int l = (DERIVED_T::IS_MULTI_SCALAR_DIFF ? 0 : k); l < (DERIVED_T::IS_MULTI_SCALAR_DIFF ? ncomp : k + 1); l++)
        {
          const int ori = DERIVED_T::IS_MULTI_SCALAR_DIFF ? (ncomp * k + l) : (DERIVED_T::IS_ANISO ? orientation(face) : k);
          const double h_imp = la_cl.h_imp(face-num1, DERIVED_T::IS_MULTI_SCALAR_DIFF ? ori : k);
          if (nu_2(j,ori) == 0.0) heq = 0.0;
          else
            {
              h_total_inv =  1.0/h_imp + e/nu_2(j,ori);
              heq = 1.0 / h_total_inv;
            }
          ajj[DERIVED_T::IS_MULTI_SCALAR_DIFF ? ncomp * k + l : k] = heq*surface(face);
          aii[DERIVED_T::IS_MULTI_SCALAR_DIFF ? ncomp * k + l : k] = is_internal ? heq*surface(face) : 0.;
        }
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::coeffs_face(const DoubleTab& inco , const int boundary_index, const int face, const int local_face, const int num1, const Echange_externe_radiatif& la_cl, Type_Double& aii, Type_Double& ajj) const
{
  assert (aii.size_array() == ajj.size_array());
  const int i = elem_(face,0), j = elem_(face,1), ncomp = DERIVED_T::IS_MULTI_SCALAR_DIFF ? int(sqrt(aii.size_array())) : aii.size_array();

  if (i != -1)
    for (int k = 0; k < ncomp; k++)
      for (int l = (DERIVED_T::IS_MULTI_SCALAR_DIFF ? 0 : k); l < (DERIVED_T::IS_MULTI_SCALAR_DIFF ? ncomp : k + 1); l++)
        {
          const int ori = DERIVED_T::IS_MULTI_SCALAR_DIFF ? (ncomp * k + l) : (DERIVED_T::IS_ANISO ? orientation(face) : k);
          const double eps = la_cl.emissivite(face-num1, DERIVED_T::IS_MULTI_SCALAR_DIFF ? ori : k);
          const double T = inco(i, l);

          aii[DERIVED_T::IS_MULTI_SCALAR_DIFF ? ncomp * k + l : k] = 4 * 5.67e-8 * eps * T * T * T * surface(face);
          ajj[DERIVED_T::IS_MULTI_SCALAR_DIFF ? ncomp * k + l : k] = 0.;
        }
  else
    for (int k = 0; k < ncomp; k++)
      for (int l = (DERIVED_T::IS_MULTI_SCALAR_DIFF ? 0 : k); l < (DERIVED_T::IS_MULTI_SCALAR_DIFF ? ncomp : k + 1); l++)
        {
          const int ori = DERIVED_T::IS_MULTI_SCALAR_DIFF ? (ncomp * k + l) : (DERIVED_T::IS_ANISO ? orientation(face) : k);
          const double eps = la_cl.emissivite(face-num1, DERIVED_T::IS_MULTI_SCALAR_DIFF ? ori : k);
          const double T = inco(j, l);

          ajj[DERIVED_T::IS_MULTI_SCALAR_DIFF ? ncomp * k + l : k] = 4 * 5.67e-8 * eps * T * T * T * surface(face);
          aii[DERIVED_T::IS_MULTI_SCALAR_DIFF ? ncomp * k + l : k] = 0.;
        }
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::coeffs_faces_interne(const int face, Type_Double& aii, Type_Double& ajj ) const
{
  assert (aii.size_array() == ajj.size_array());
  const int i = elem_(face,0), j = elem_(face,1), ncomp = DERIVED_T::IS_MULTI_SCALAR_DIFF ? int(sqrt(aii.size_array())) : aii.size_array();
  double heq, d0 = Dist_face_elem0(face,i), d1 = Dist_face_elem1(face,j);
  for (int k = 0; k < ncomp; k++)
    {
      const int ori = DERIVED_T::IS_ANISO ? orientation(face) : k;
      if (DERIVED_T::IS_RANS)
        {
          heq = compute_heq(d0,i,d1,j,ori);
          aii[k] = ajj[k] = heq*surface(face)*porosite(face); // On peut faire ca !
        }
      else if (DERIVED_T::IS_MULTI_SCALAR_DIFF)
        {
          for (int l = 0; l < ncomp; l++)
            {
              heq = compute_heq(d0, i, d1, j, ncomp * k + l);
              aii[ncomp * k + l] = heq * surface(face) * porosite(face);
              ajj[ncomp * k + l] = heq * surface(face) * porosite(face);
            }
        }
      else
        {
          if (nu_1(i,ori) == 0.0 && nu_2(j,ori) == 0.0) heq = 0.;
          else
            {
              assert(nu_1(i,ori) != 0.0 && nu_2(j,ori) != 0.0);
              heq = compute_heq(d0,i,d1,j,ori);
            }
          aii[k] = ajj[k] = heq*surface(face)*porosite(face); // On peut faire ca !
        }
    }
}

///////////////////////////////////////////////////////////////
// A virer un jour .. voir avec le baltik Rayonnement
template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::secmem_face(const int face, const Dirichlet_entree_fluide& la_cl, const int num1, Type_Double& flux) const
{
  const int i = elem_(face,0), j = elem_(face,1), ncomp = flux.size_array();
  double dist = Dist_norm_bord(face);
  for (int k = 0; k < ncomp; k++)
    {
      const int ori = DERIVED_T::IS_ANISO ? orientation(face) : k;
      const double T_imp = la_cl.val_imp(face-num1,k);
      flux[k] = (i != -1) ? T_imp*surface(face)*porosite(face)*nu_1(i,ori) / dist : -T_imp*surface(face)*(porosite(face)*nu_1(j,ori)) / dist;
    }
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::secmem_face(const int face, const Neumann_paroi& la_cl, const int num1, Type_Double& flux) const
{
  const int i = elem_(face,0), ncomp = flux.size_array();
  for (int k = 0; k < ncomp; k++) flux[k] = ((i != -1) ? 1 : -1) * la_cl.flux_impose(face-num1,k)*surface(face);
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::secmem_face(const int face, const Echange_global_impose& la_cl, const int num1, Type_Double& flux) const
{
  const int i = elem_(face,0), ncomp = flux.size_array();
  for (int k = 0; k < ncomp; k++)
    {
      const double h = la_cl.h_imp(face-num1,k), T_ext = la_cl.T_ext(face-num1,k);
      const double phi_ext = la_cl.has_phi_ext() ? la_cl.flux_exterieur_impose(face-num1, k) : 0.;
      flux[k] = ((i != -1) ? 1.0 : -1.0) * (phi_ext + h * T_ext) * surface(face);
    }
}

template <typename DERIVED_T> template <typename Type_Double>
inline void Eval_Diff_VDF_Elem_Gen<DERIVED_T>::secmem_face(const int boundary_index, const int face, const int local_face, const Echange_externe_impose& la_cl, const int num1, Type_Double& flux) const
{
  // C.L de type Echange_externe_impose : 1/h_total = (1/h_imp) + (e/diffusivite) : La C.L fournit h_imp ; il faut calculer e/diffusivite
  double e, h_total_inv, heq;
  const int i = elem_(face,0), j = elem_(face,1), ncomp = flux.size_array();

  if (DERIVED_T::IS_MODIF_DEQ) e = (ind_Fluctu_Term() == 1) ? Dist_norm_bord_externe_(face) : equivalent_distance(boundary_index,local_face);
  else e = DERIVED_T::IS_DEQUIV ? equivalent_distance(boundary_index,local_face) : Dist_norm_bord_externe_(face);

  if (i != -1)
    {
      for (int k = 0; k < ncomp; k++)
        {
          const int ori = DERIVED_T::IS_ANISO ? orientation(face) : k;
          const double h_imp = la_cl.h_imp(face-num1,k), T_ext = la_cl.T_ext(face-num1, k);
          if (nu_2(i,ori) == 0.0) heq = 0.0;
          else
            {
              h_total_inv = 1.0/h_imp + e/nu_2(i,ori);
              heq = 1.0 / h_total_inv;
            }
          flux[k] = heq*T_ext*surface(face);
        }
    }
  else
    {
      for (int k = 0; k < ncomp; k++)
        {
          const int ori = DERIVED_T::IS_ANISO ? orientation(face) : k;
          const double h_imp = la_cl.h_imp(face-num1,k), T_ext = la_cl.T_ext(face-num1, k);
          if (nu_2(j,ori) == 0.0) heq = 0.0;
          else
            {
              h_total_inv = 1.0/h_imp + e/nu_2(j,ori);
              heq = 1.0 / h_total_inv;
            }
          flux[k] = -heq*T_ext*surface(face);
        }
    }
}

#endif /* Eval_Diff_VDF_Elem_Gen_TPP_included */
