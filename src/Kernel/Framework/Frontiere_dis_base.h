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

#ifndef Frontiere_dis_base_included
#define Frontiere_dis_base_included

#include <TRUST_Ref.h>
#include <Nom.h>
#include <Domaine_forward.h>

class Domaine_dis_base;

/*! @brief classe Frontiere_dis_base Classe representant une frontiere discretisee.
 *
 * Les objets
 *      Frontiere_dis_base contiennent un reference vers l'objet
 *      geometrique Frontiere.
 *
 * @sa Frontiere Domaine_dis
 */
class Frontiere_dis_base : public Objet_U
{
  Declare_base(Frontiere_dis_base);

public :

  void associer_frontiere(const Frontiere&);
  void associer_Domaine_dis(const Domaine_dis_base&);
  const Frontiere& frontiere() const;
  Frontiere& frontiere();
  const Domaine_dis_base& domaine_dis() const;
  const Nom& le_nom() const override;

protected :

  OBS_PTR(Frontiere) la_frontiere;
  OBS_PTR(Domaine_dis_base) le_dom_dis;
};

#endif
