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

#include <Discretisation_base.h>
#include <Paroi_flux_impose.h>
#include <Equation_base.h>

Implemente_instanciable(Paroi_flux_impose, "paroi_flux_impose", Cond_lim_utilisateur_base);
// XD paroi_flux_impose condlim_base paroi_flux_impose 0 Normal flux condition at the wall called bord (edge). The surface area of the flux (W.m-1 in 2D or W.m-2 in 3D) is imposed at the boundary according to the following convention: a positive flux is a flux that enters into the domain according to convention.
// XD attr ch front_field_base ch 0 Boundary field type.

Sortie& Paroi_flux_impose::printOn(Sortie& s ) const { return s << que_suis_je() ; }

Entree& Paroi_flux_impose::readOn(Entree& s ) { return s; }

void Paroi_flux_impose::complement(Nom& ajout)
{
  int rayo = is_pb_rayo();
  if (rayo == 1)
    {
      if (mon_equation->discretisation().is_vdf())
        ajout = "paroi_flux_impose_rayo_semi_transp_VDF";
      else
        ajout = "paroi_flux_impose_rayo_semi_transp_VEF";
    }
  else if (rayo == 2)
    ajout = "paroi_flux_impose_rayo_transp";
  else
    ajout = "Neumann_Paroi";
}
