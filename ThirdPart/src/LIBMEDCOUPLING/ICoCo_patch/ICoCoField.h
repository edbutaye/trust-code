// Copyright (C) 2007-2019  CEA/DEN, EDF R&D
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
//
// See http://www.salome-platform.org/ or email : webmaster.salome@opencascade.com
//

// WARNING: this file is part of the official ICoCo API and should not be modified.
// The official version can be found in the TRUST repository:
//
//    https://sourceforge.net/projects/trust/

#ifndef _ICoCoField_included_
#define _ICoCoField_included_

#include <string>

namespace ICoCo {

  class Field {
  public:
    Field();
    virtual ~Field();
    void setName(const std::string& name);
    const std::string& getName() const;
    const char* getCharName() const;

  private:
    std::string* _name;
  };
}
#endif
