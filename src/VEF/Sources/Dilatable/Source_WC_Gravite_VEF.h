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

#ifndef Source_WC_Gravite_VEF_included
#define Source_WC_Gravite_VEF_included

#include <Source_Gravite_Fluide_Dilatable_base.h>
#include <Source_Fluide_Dilatable_VEF_Proto.h>

/*! @brief class  Source_WC_Gravite_VEF
 *
 *  Cette classe represente un terme source supplementaire
 *  a prendre en compte dans les equations de quantite de mouvement
 *   dans le cas ou le fluide est weakly compressible, en cas de gravite, et pour
 *   une discretisation VEF.
 *
 *
 * @sa Source_base Fluide_Weakly_Compressible Source_Gravite_Fluide_Dilatable_base
 */

class Source_WC_Gravite_VEF : public Source_Gravite_Fluide_Dilatable_base,
  public Source_Fluide_Dilatable_VEF_Proto
{
  Declare_instanciable(Source_WC_Gravite_VEF);

public:
  DoubleTab& ajouter(DoubleTab& ) const override;

protected :
  void associer_domaines(const Domaine_dis_base& domaine,const Domaine_Cl_dis_base& ) override;
};

#endif /* Source_WC_Gravite_VEF_included */
