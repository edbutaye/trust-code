/****************************************************************************
* Copyright (c) 2019, CEA
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
// File:        Format_Post_XYZ.h
// Directory:   $TRUST_ROOT/src/Kernel/Framework
// Version:     /main/13
//
//////////////////////////////////////////////////////////////////////////////
#ifndef Format_Post_XYZ_included
#define Format_Post_XYZ_included

#include <Format_Post_base.h>
#include <EcrFicPartage.h>

class DoubleTab;
class IntTab;

// .DESCRIPTION        :
//  Classe de postraitement des champs euleriens au format xyz.

///////////////////////////////////////////////////////////////////////
//    Pour creer un fichier valide, il faut faire:
//    (initialisation) initialize("base_nom_fichier",...);
//                     //base_nom_fichier indique le nom du fichier sans extension ".xyz"
//                     //met eventuellement en jeux les informations d autres postraitements
//                     ecrire_entete(temps_courant,reprise,est_le_premier_post)
//                     ecrire_domaine(domaine,est_le_premier_post)
//
//    (pour chaque dt) ecrire_temps(temps_courant)
//                     ecrire_champ(const Domaine& domaine, const Noms& unite_, const Noms& noms_compo,
//                                      int ncomp,double temps_,double temps_courant
//                                      const Nom  & id_du_champ,
//                                    const Nom  & id_du_domaine,
//                                    const Nom  & localisation,
//                                    const DoubleTab & data)
//                    [ecrire_champ(..., data)]
//
//    (finir)          finir(est_le_dernier_post)
//////////////////////////////////////////////////////////////////////

class Format_Post_XYZ : public Format_Post_base
{
  Declare_instanciable_sans_constructeur(Format_Post_XYZ);
public:
  // Methodes declarees dans la classe de base (interface commune a tous
  // les formats de postraitment de champs):
  virtual void reset();
  void set_param(Param& param);
  virtual int initialize_by_default(const Nom& file_basename);
  virtual int initialize(const Nom& file_basename, const int& format, const Nom& option_para);
  virtual int ecrire_entete(double temps_courant,int reprise,const int& est_le_premier_post);
  virtual int completer_post(const Domaine& dom,const int axi,
                             const Nature_du_champ& nature,const int nb_compo,const Noms& noms_compo,
                             const Motcle& loc_post,const Nom& le_nom_champ_post);

  virtual int preparer_post(const Nom& id_du_domaine,const int& est_le_premier_post,
                            const int& reprise,
                            const double& t_init);
  virtual int finir(int& est_le_dernier_post);
  virtual int ecrire_domaine(const Domaine& domaine,const int& est_le_premie_post);
  virtual int ecrire_temps(const double temps);

  virtual int ecrire_item_int(const Nom&   id_item,
                              const Nom&   id_du_domaine,
                              const Nom&   id_zone,
                              const Nom&   localisation,
                              const Nom&   reference,
                              const IntVect& data,
                              const int reference_size);

  virtual int ecrire_champ2(const Domaine& domaine, const Noms& unite_, const Noms& noms_compo,
                            int ncomp,double temps_,double temps_courant,
                            const Nom&   id_du_champ,
                            const Nom&   id_du_domaine,
                            const Nom&   localisation,
                            const Nom&   nature,
                            const DoubleTab& data,
                            const DoubleTab& coord);

  // Methodes statiques specifiques a ce format:

  Nom construire_nom_xyz(double temps_, const Nom& id_champ,
                         const Nom& id_du_domaine,const Nom& localisation);
  static int ecrire_domaine_xyz(const Domaine& domaine,Nom& nom_fic);
  static int ecrire_temps_xyz(const double temps,Nom& nom_fic);

  static int ecrire_champ_xyz(const Domaine& domaine,const Noms& unite_, const Noms& noms_compo,
                              int ncomp, double temps_,
                              const Nom&   id_du_champ,
                              const Nom&   id_du_domaine,
                              const Nom&   localisation,
                              const DoubleTab& data,Nom& nom_fic,
                              const DoubleTab& coord);

  static int ecrire_item_int_xyz(const Nom&   id_item,
                                 const Nom&   id_zone,
                                 const IntVect& data,
                                 const Nom& nom_fic);

  static int ecrire_entete_xyz(Nom& nom_fic,const int& est_le_premier_post);
  static int finir_xyz(Nom& nom_fic, const int& est_le_dernier_post);
  static int completer_post_xyz();
  static int preparer_post_xyz();

  Format_Post_XYZ();

protected:

  Nom  xyz_basename_;

public:

};

#endif
