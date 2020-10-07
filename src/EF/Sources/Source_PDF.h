//////////////////////////////////////////////////////////////////////////////
//
// File:        Source_PDF.h
// Directory:   $TRUST_ROOT/EF/Sources
// Version:     /main/2
//
//////////////////////////////////////////////////////////////////////////////


#ifndef Source_PDF_included
#define Source_PDF_included

#include <Source_PDF_base.h>
#include <Ref_Zone_EF.h>
#include <Ref_Zone_Cl_EF.h>
#include <Champ_Fonc.h>
#include <Matrice.h>

class Probleme_base;

//
// .DESCRIPTION class Source_PDF
//  Source term for the penalisation of the momentum in the Immersed Boundary Method (IBM)
//
// .SECTION voir aussi Source_PDF_base
//
//

class Source_PDF : public Source_PDF_base
{

  Declare_instanciable(Source_PDF);

public:
  void associer_pb(const Probleme_base& );
  DoubleTab& ajouter_(const DoubleTab& vitesse, DoubleTab&) const;
  void contribuer_a_avec(const DoubleTab&, Matrice_Morse&) const;
  void verif_ajouter_contrib(const DoubleTab& vitesse, Matrice_Morse& matrice) const ;
  DoubleVect diag_coeff_elem(ArrOfDouble&, const DoubleTab&, int) const ;
  DoubleTab compute_coeff_elem() const;
  DoubleTab compute_coeff_matrice_pression() const;
  void multiply_coeff_volume(DoubleTab&) const;
  void correct_pressure(const DoubleTab&,DoubleTab&,const DoubleTab&) const;
  void correct_incr_pressure(const DoubleTab&,DoubleTab&) const;
  void correct_vitesse(const DoubleTab&,DoubleTab&) const;
  void calculer_vitesse_imposee();
  void test(Matrice&) const;
  void updateChampRho();
  int impr(Sortie&) const;
protected:
  void compute_vitesse_imposee_projete(const DoubleTab&, const DoubleTab&, double, double);
  void calculer_vitesse_imposee_hybrid();
  void calculer_vitesse_imposee_elem_fluid();
  void calculer_vitesse_imposee_mean_grad();
  void rotate_imposed_velocity(DoubleTab&);
  DoubleTab compute_pond(const DoubleTab&, const DoubleTab&, const DoubleVect&, int&, int&) const ;
  REF(Zone_EF) la_zone_EF;
  REF(Zone_Cl_EF) la_zone_Cl_EF;
  void associer_zones(const Zone_dis& ,const Zone_Cl_dis& );
};

#endif
