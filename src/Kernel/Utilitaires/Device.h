/****************************************************************************
* Copyright (c) 2022, CEA
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
// File:        Device.h
// Directory:   $TRUST_ROOT/src/Kernel/Utilitaires
// Version:     /main/16
//
//////////////////////////////////////////////////////////////////////////////
#ifndef Device_included
#define Device_included

// Timer:
#include <ctime>
#include <string>
#include <Array_base.h>
static std::clock_t clock_start;
static char* clock_on=NULL;
inline void start_timer()
{
  clock_on = getenv ("TRUST_CLOCK_ON");
  if (clock_on!=NULL) clock_start = std::clock();
}
inline void end_timer(const std::string& str) // Return in [ms]
{
  if (clock_on!=NULL)
    {
      printf("[clock] %7.3f ms %s\n", 1000*(std::clock() - clock_start) / (double) CLOCKS_PER_SEC ,str.c_str());
      fflush(stdout);
    }
}
inline void end_timer(const std::string& str, int size) // Return in [ms]
{
  if (clock_on!=NULL)
    {
      double ms = 1000*(std::clock() - clock_start) / (double) CLOCKS_PER_SEC;
      int mo = size/1024/1024;
      printf("[clock] %7.3f ms %s %6d Mo %4.1f Go/s\n", ms ,str.c_str(), mo, mo/ms);
      fflush(stdout);
    }
}

template <typename _TYPE_>
inline const _TYPE_* copyToDevice(const TRUSTArray<_TYPE_>& tab)
{
  const _TYPE_ *tab_addr = copyToDevice_(const_cast<TRUSTArray <_TYPE_>&>(tab));
  tab.set_dataLocation(Array_base::HostDevice); // const array will matches on host and device
  return tab_addr;
}
/* ToDo conflit possible avec const _TYPE_* copyToDevice(TRUSTArray<_TYPE_>& tab)
 // Appeler computeOnDevice ?
template <typename _TYPE_>
inline _TYPE_* copyToDevice(TRUSTArray<_TYPE_>& tab)
{
  _TYPE_ *tab_addr = copyToDevice_(tab);
  tab.set_dataLocation(Array_base::Device); // non-const array will be modified on device
  return tab_addr;
} */
template <typename _TYPE_>
inline _TYPE_* copyToDevice_(TRUSTArray<_TYPE_>& tab)
{
  _TYPE_* tab_addr = tab.addr();
#ifdef _OPENMP
  start_timer();
  if (tab.dataLocation()==Array_base::HostOnly)
    {
      #pragma omp target enter data map(to:tab_addr[0:tab.size_array()])
      end_timer((std::string) "copyToDevice Array ", sizeof(_TYPE_) * tab.size_array());
    }
  else if (tab.dataLocation()==Array_base::Host)
    {
      #pragma omp target update to(tab_addr[0:tab.size_array()])
      end_timer((std::string) "updateToDevice Array ", sizeof(_TYPE_) * tab.size_array());
    }
  else
    end_timer((std::string) "presentOnDevice Array ", sizeof(_TYPE_) * tab.size_array());
#endif
  return tab_addr;
}
#endif
