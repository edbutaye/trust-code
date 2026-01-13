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

#pragma GCC diagnostic push

#if __GNUC__ < 9
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif

#include <gtest/gtest.h>

#pragma GCC diagnostic pop

#include <TRUSTTrav.h>
#include <TRUSTTab_parts.h>
#include <Device.h>

#ifndef TRUST_GTEST
#define TRUST_GTEST 
#endif

#include <DeviceMemory.h>

#ifdef __NVCOMPILER
#pragma diag_suppress 177
#endif

#ifdef TRUST_USE_GPU 

TEST(DeviceTest, KokkosReductionBug) {
    DoubleTab a(2);
    a(0) = 1;
    a(1) = 2;
    mapToDevice(a);
    double mp = mp_norme_vect(a);
    double sol = sqrt(Process::nproc() * 5);
    if (mp != sol)
    {
          Cerr << "What! " << mp << " != " << sol << finl;
          Cerr
              << "Weird bug seen during a sum reduction with Kokkos::parallel_reduce for TRUST production build on Cuda 11.6. Fixed with Cuda 12.1 (or before...)."
              << finl;
          Cerr << "Update Cuda please !" << finl;
        }
    EXPECT_EQ(mp, sol);
}

TEST(DeviceTest, MapToDevice)
{
  DoubleTab a(10);
  EXPECT_EQ(a.get_data_location(), DataLocation::HostOnly);
  mapToDevice(a);
  EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
  a[1] = 0;
  EXPECT_EQ(a.get_data_location(), DataLocation::Host);
  mapToDevice(a);
  EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
}

TEST(DeviceTest, MapToDeviceOnARef)
{
  DoubleTab a(10);
  a = 23;
  EXPECT_EQ(a.get_data_location(), DataLocation::HostOnly);
  {
    DoubleTab b;
    b.ref(a);
    mapToDevice(b);
    EXPECT_EQ(b.get_data_location(), DataLocation::HostDevice);
    EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
  }
}

TEST(DeviceTest, AccessMethod0)
{
  int N = 10;
  DoubleTab inco(N);
  inco = 1;
  mapToDevice(inco); // copy
  EXPECT_EQ(inco.get_data_location(), DataLocation::HostDevice);
  EXPECT_EQ(inco.ref_count(), 1);

  {
    // Exemple 1er operateur
    DoubleTab a;
    a.ref(inco); // Doit prendre l'etat de inco
    EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
    EXPECT_EQ(a.ref_count(), 2);
    EXPECT_EQ(inco.ref_count(), 2);

    DoubleTab b(N);
    CDoubleArrView a_v = static_cast<const ArrOfDouble&>(a).view_ro();
    DoubleArrView b_v = static_cast<ArrOfDouble&>(b).view_wo();

    Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i) {
      b_v[i] = a_v[i];
    });
    Kokkos::fence();

    const DoubleTab& const_a = a;
    const DoubleTab& const_b = b;
    EXPECT_EQ(const_b[5], const_a[5]);
    EXPECT_EQ(const_b[5], 1);
    //assert(b[5] == a[5]); // Argh double& TRUSTArray<double>::operator[](int i) appele pour a et donc repasse sur host
    // Comment detecter que operator[](int i) est utilise en read ou write ? Possible ? Non, sauf si const utilise.
    EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
    EXPECT_EQ(b.get_data_location(), DataLocation::HostDevice);
    EXPECT_EQ(inco.get_data_location(), DataLocation::HostDevice);
  }
  EXPECT_EQ(inco.get_data_location(), DataLocation::HostDevice);
  EXPECT_EQ(inco.ref_count(), 1);
}

TEST(DeviceTest, AccessMethod1)
{
  int N = 10;
  DoubleTab inco(N);
  inco = 1;
  mapToDevice(inco);
  EXPECT_EQ(inco.get_data_location(), DataLocation::HostDevice);
  EXPECT_EQ(inco.ref_count(), 1);

  {
    DoubleTab a;
    a.ref(inco);
    EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
    EXPECT_EQ(a.ref_count(), 2);
    EXPECT_EQ(inco.ref_count(), 2);

    DoubleTab b(N);
    CDoubleArrView a_v = static_cast<const ArrOfDouble&>(a).view_ro();
    DoubleArrView b_v = static_cast<ArrOfDouble&>(b).view_wo();

    Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i) {
      b_v[i] = a_v[i];
    });
    Kokkos::fence();

    const DoubleTab& const_a = a;
    const DoubleTab& const_b = b;
    EXPECT_EQ(const_b[5], const_a[5]);
    EXPECT_EQ(const_b[5], 1);
    EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
    EXPECT_EQ(b.get_data_location(), DataLocation::HostDevice);
  }
  EXPECT_EQ(inco.ref_count(), 1);
}

TEST(DeviceTest, UpdateOnDevice0)
{
  int N = 10;
  DoubleTab inco(N);
  inco = 1;
  mapToDevice(inco);
  EXPECT_EQ(inco.get_data_location(), DataLocation::HostDevice);
  EXPECT_EQ(inco.ref_count(), 1);

  // Mise à jour sur le device
  inco += 1;
  EXPECT_EQ(inco.get_data_location(), DataLocation::Device);
  EXPECT_EQ(inco.ref_count(), 1);

  {
    DoubleTab a;
    a.ref(inco);
    EXPECT_EQ(a.get_data_location(), DataLocation::Device);
    EXPECT_EQ(a.ref_count(), 2);
    EXPECT_EQ(inco.ref_count(), 2);

    DoubleTab b(N);
    CDoubleArrView a_v = static_cast<const ArrOfDouble&>(a).view_ro();
    DoubleArrView b_v = static_cast<ArrOfDouble&>(b).view_wo();

    Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i) {
      b_v[i] = a_v[i];
    });
    Kokkos::fence();

    const DoubleTab& const_a = a;
    const DoubleTab& const_b = b;
    EXPECT_EQ(const_b[5], const_a[5]);
    EXPECT_EQ(const_b[5], 2);
    EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
    EXPECT_EQ(b.get_data_location(), DataLocation::HostDevice);
    EXPECT_EQ(inco.get_data_location(), DataLocation::HostDevice); // Car a ref sur inco
  }
  EXPECT_EQ(inco.get_data_location(), DataLocation::HostDevice);

  // Deuxième mise à jour : inco passe à 3
  inco += 1;
  EXPECT_EQ(inco.get_data_location(), DataLocation::Device);

  {
    DoubleTab a;
    a.ref(inco);
    EXPECT_EQ(a.get_data_location(), DataLocation::Device);
    EXPECT_EQ(a.ref_count(), 2);
    EXPECT_EQ(inco.ref_count(), 2);

    DoubleTab b(N);
    CDoubleArrView a_v = static_cast<const ArrOfDouble&>(a).view_ro();
    DoubleArrView b_v = static_cast<ArrOfDouble&>(b).view_wo();

    Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i) {
      b_v[i] = a_v[i];
    });
    Kokkos::fence();

    const DoubleTab& const_a = a;
    const DoubleTab& const_b = b;
    EXPECT_EQ(const_b[5], const_a[5]);
    EXPECT_EQ(const_b[5], 3);
    EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
    EXPECT_EQ(b.get_data_location(), DataLocation::HostDevice);
    EXPECT_EQ(inco.get_data_location(), DataLocation::HostDevice);
  }
  EXPECT_EQ(inco.get_data_location(), DataLocation::HostDevice);
}

TEST(DeviceTest, OperationsArrOfDouble)
{
  ArrOfDouble a(10), b(10);
  a = 1;
  b = 2;
  mapToDevice(a);
  mapToDevice(b);

  // Opérations réalisées sur le device
  b+=a; // TRUSTArray& operator+=(const TRUSTArray& y) sur le device
  b+=3; // TRUSTArray& operator+=(const _TYPE_ dy)
  b-=2; // TRUSTArray& operator-=(const _TYPE_ dy)
  b-=a; // TRUSTArray& operator-=(const TRUSTArray& y)
  EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
  EXPECT_EQ(b.get_data_location(), DataLocation::Device);
  const ArrOfDouble& const_b = b;

  // Retour sur le host pour vérifier le résultat
  copyFromDevice(b);
  EXPECT_EQ(const_b[0], 3);
}

TEST(DeviceTest, allocateOnDevice)
{
  DoubleTab a(10);
  allocateOnDevice(a);
  EXPECT_EQ(a.get_data_location(), DataLocation::Device);
  a = 1; // Affectation sur le device
  EXPECT_EQ(a.get_data_location(), DataLocation::Device);

  // Constructeur par copie : la copie doit être faite sur le device
  DoubleTab b(a);
  EXPECT_EQ(b.get_data_location(), DataLocation::Device);
  copyFromDevice(b);
  const ArrOfDouble& const_b = b;
  EXPECT_EQ(const_b[0], 1);
  EXPECT_EQ(const_b[b.size() - 1], 1);
}

TEST(DeviceTest, copyFromDevice)
{
  DoubleTab a(10);
  a = 1;
  mapToDevice(a); // a est sur le device
  DoubleTab b;
  b = a; // copie de a sur le device
  EXPECT_EQ(b.get_data_location(), DataLocation::Device);
  const ArrOfDouble& const_b = b;
  copyFromDevice(b);
  EXPECT_EQ(const_b[0], 1);
  EXPECT_EQ(const_b[b.size() - 1], 1);
}

TEST(DeviceTest, OperatorTest)
{
  DoubleTab a(10), b(10);
  const ArrOfDouble& const_a = a;
  const ArrOfDouble& const_b = b;
  a = 1;
  b = 10;
  mapToDevice(a);
  EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
  mapToDevice(b);
  EXPECT_EQ(b.get_data_location(), DataLocation::HostDevice);

  a = b;  // Injection de a par copie sur device (a devient 10)
  EXPECT_EQ(a.get_data_location(), DataLocation::Device);
  a += b; // a devient 20 sur device
  EXPECT_EQ(a.get_data_location(), DataLocation::Device);
  b -= a; // b devient -10 sur device
  EXPECT_EQ(b.get_data_location(), DataLocation::Device);

  // Retour sur host pour vérification
  copyFromDevice(a);
  copyFromDevice(b);
  EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
  EXPECT_EQ(b.get_data_location(), DataLocation::HostDevice);
  EXPECT_EQ(const_a[0], 20);
  EXPECT_EQ(const_a[a.size() - 1], 20);
  EXPECT_EQ(const_b[0], -10);
  EXPECT_EQ(const_b[b.size() - 1], -10);
}

TEST(DeviceTest, DeviceDoubleTrav0)
{
  int N = 10;
  DoubleTrav a(10 * N);
  a = 1;
  mapToDevice(a); // copie sur le device
  EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
  EXPECT_EQ(a.ref_count(), 1);
  DeviceMemory::printMemoryMap();
}

TEST(DeviceTest, DeviceDoubleTrav1)
{
  int N = 10;
  DeviceMemory::printMemoryMap();
  DoubleTrav a(10 * N); // doit être initialisé à 0 sur le device
  EXPECT_EQ(a.get_data_location(), DataLocation::Device);
  const ArrOfDouble& const_a = a;
  EXPECT_EQ(const_a[0], 0);
}

TEST(DeviceTest, DoubleTabCopyConstructorTest)
{
  int N = 10;
  DoubleTab a(N);
  a = -1;
  mapToDevice(a); // a sur le device
  DoubleTrav b(a); // b doit être sur le device
  EXPECT_EQ(b.get_data_location(), DataLocation::Device);
  b += 1; // opération sur le device
  EXPECT_EQ(b.get_data_location(), DataLocation::Device);
  copyFromDevice(b);
  const ArrOfDouble& const_b = b;
  EXPECT_EQ(const_b[0], 1);
  EXPECT_EQ(const_b[b.size() - 1], 1);
}

TEST(DeviceTest, MinMaxTest)
{
  DoubleTab a(3);
  a(0) = 1;
  a(1) = 3;
  a(2) = -10;
  mapToDevice(a);

  // Modification sur le host pour le test
  a.data()[0] = 0;
  a.data()[1] = 0;
  a.data()[2] = 0;
  a.set_data_location(DataLocation::Device);

  EXPECT_EQ(local_max_vect(a), 3);
  EXPECT_EQ(local_min_vect(a), -10);
  EXPECT_EQ(local_max_abs_vect(a), 10);
  EXPECT_EQ(local_min_abs_vect(a), 1);
  // Les méthodes local_imax_vect et local_imin_vect ne sont pas portées sur GPU :
  EXPECT_TRUE(a.isDataOnDevice());

  // Retour sur host pour tester imax/imin
  copyFromDevice(a);
  EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
  EXPECT_EQ(local_imax_vect(a), 1);
  EXPECT_EQ(local_imin_vect(a), 2);
}

TEST(DeviceTest, RefArrayTest)
{
  int N = 10;
  DoubleTab a(N);
  a = 1;
  mapToDevice(a);
  EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
  DoubleTab b;
  b.ref_array(a);
  EXPECT_EQ(b.get_data_location(), DataLocation::HostDevice);
}

TEST(DeviceTest, RefTabChunkTest)
{
  int N = 10;
  DoubleTab a(2 * N);
  a = 1;
  DoubleTab b;
  b.ref_tab(a, 0, N); // référence partielle sur a
  mapToDevice(b);
  EXPECT_EQ(b.get_data_location(), DataLocation::HostDevice);
  EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);

  DoubleArrView a_v = static_cast<ArrOfDouble&>(a).view_wo();
  Kokkos::parallel_for(2 * N, KOKKOS_LAMBDA(const int i) {
    a_v[i] = 2;
  });
  Kokkos::fence();

  // Après lecture, a doit être entièrement sur le device, puis basculer sur le host
  EXPECT_EQ(a.get_data_location(), DataLocation::Device);
  EXPECT_EQ(a(0), 2);
  EXPECT_EQ(a.get_data_location(), DataLocation::Host);
  EXPECT_EQ(a(2 * N - 1), 2);
}

TEST(DeviceTest, HostPointerTest)
{
  int N = 10;
  double* ptr_host = nullptr;
  {
    DoubleTab a(N);
    a = 1;
    {
      DoubleTab b;
      b.ref_array(a);
      mapToDevice(b);
      EXPECT_EQ(a.data(), b.data());
      EXPECT_EQ(b.get_data_location(), DataLocation::HostDevice);
      EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
    }
    EXPECT_EQ(a.get_data_location(), DataLocation::HostDevice);
    ptr_host = a.data();
    EXPECT_TRUE(isAllocatedOnDevice(ptr_host));
  }
  EXPECT_FALSE(isAllocatedOnDevice(ptr_host));
}

TEST(DeviceTest, Resize)
{
  DoubleTab a(1);
  mapToDevice(a);
  a = 1;
  a.resize(2);
  a += 1;
  EXPECT_TRUE(a.isDataOnDevice());
  EXPECT_EQ(a(0), 2); // L'ancien élément est conservé (1+1)
  EXPECT_EQ(a(1), 1); // Le nouvel élément est initialisé à 0 et ensuite incrémenté de 1
}

TEST(DeviceTest, DoubleTravCopyConstructor)
{
  DoubleTrav b(100);
  b = 123;
  mapToDevice(b);
  DoubleVect b2(b);
  EXPECT_EQ(b2.get_mem_storage(), STORAGE::TEMP_STORAGE);
  EXPECT_EQ(b2.get_data_location(), DataLocation::Device);

  DoubleVect b3(b2);
  EXPECT_EQ(b3.get_mem_storage(), STORAGE::TEMP_STORAGE);
  EXPECT_EQ(b3.get_data_location(), DataLocation::Device);
}



TEST(DeviceTest, copyToDevice2)
{

  int n0=2, n1=3, n2=4;
  TRUSTTab<int> tab(n0,n1,n2);

  for (int i=0; i<n0; i++){
    for (int j=0; j<n1; j++){
      for (int k=0; k<n2; k++){
          tab(i,j,k) = i+j-k;
      }
    }
  }

  //This does a map to device
  auto view_rw = tab.view_rw<3, Kokkos::DefaultExecutionSpace>();
  
  // Parallel reduce to check if all values match i + j + k
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n0, n1, n2});

  bool all_correct = true;
  Kokkos::parallel_reduce("CheckValues", policy,
                          KOKKOS_LAMBDA(int i, int j, int k, bool& result) {
                    
                            if (view_rw(i, j, k) != i + j - k) {result = false;printf("Mismatch at i=%d, j=%d, k=%d: view_rw(i,j,k)=%d, expected=%d\n", 
           i, j, k, view_rw(i, j, k), i + j - k);}},
          Kokkos::LAnd<bool>(all_correct));

  EXPECT_TRUE(all_correct);
}

TEST(DeviceTest, copyFromDevice2)
{

  int n0=10, n1=11, n2=12;
  TRUSTTab<int> tab(n0,n1,n2);

  //This does a map to device
  auto view_rw = tab.view_rw<3, Kokkos::DefaultExecutionSpace>();

  // Parallel reduce to check if all values match i + j + k
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n0, n1, n2});

  Kokkos::parallel_for("SetValues", policy,
                      KOKKOS_LAMBDA(int i, int j, int k) 
                      {
                        view_rw(i, j, k) = i + j - k;
                      });

  //copyFromDevice(tab);

  for (int i=0; i<n0; i++){
    for (int j=0; j<n1; j++){
      for (int k=0; k<n2; k++){
          EXPECT_EQ(i+j-k, tab(i,j,k));
      }
    }
  }
}

TEST(DeviceTest, resizeGPUArrayUP){

  int N=10;
  TRUSTArray<int> a(N);

  for (int i=0; i<N; i++){a[i]=i;}
  auto view_rw = a.view_rw();

  a.resize(2*N);

  for (int i=0; i<N; i++){EXPECT_EQ(i, a[i]);}

  auto policy = Kokkos::RangePolicy(0, N);

  bool all_correct = true;
  Kokkos::parallel_reduce("CheckValues", policy,
                          KOKKOS_LAMBDA(int i, bool& result) {
                            if (view_rw(i) != i) {result = false;printf("Mismatch at i=%d, view_rw(i)=%d\n", i, view_rw(i));}},
          Kokkos::LAnd<bool>(all_correct));
  EXPECT_TRUE(all_correct);
}

TEST(DeviceTest, resizeGPUArrayDOWN){

  int N=10;
  TRUSTArray<double> a(N);
  for (int i=0; i<N; i++){a[i]=i;}
  auto view_rw = a.view_rw();

  a.resize(N/2);
  for (int i=0; i<N/2; i++){EXPECT_EQ(i, a[i]);}

  auto policy = Kokkos::RangePolicy(0, N/2);

  bool all_correct = true;
  Kokkos::parallel_reduce("CheckValues", policy,
                          KOKKOS_LAMBDA(int i, bool& result) {
                            if (view_rw(i) != i) {result = false;printf("Mismatch at i=%d, view_rw(i)=%f\n", i, view_rw(i));}},
          Kokkos::LAnd<bool>(all_correct));
  EXPECT_TRUE(all_correct);
}


TEST(DeviceTest, resizeGPUTabUP){

  TRUSTTab<int> tab(2, 3);

  tab(0,0)=0;
  tab(0,1)=1;
  tab(0,2)=2;
  tab(1,0)=3;
  tab(1,1)=4;
  tab(1,2)=5;

  auto view = tab.view_rw();

  bool all_correct = true;
  Kokkos::parallel_reduce("CheckValues", 6,
                          KOKKOS_LAMBDA(int i, bool& result) {
                            int k=(int)(i>=3);
                            int l=i%3;
                            if (view(k,l) != i) {result = false;
                             printf("Mismatch, i=%d, view(%d,%d)=%d\n", i, k,l, view(k,l));}},
          Kokkos::LAnd<bool>(all_correct));
  EXPECT_TRUE(all_correct);
  
  tab.resize(4,6);

  auto view2 = tab.view_rw();

  for (int i=0; i<6; i++){
      EXPECT_EQ(tab(0,i),i);
  }
  
  all_correct = true;
  Kokkos::parallel_reduce("CheckValues", 6,
                          KOKKOS_LAMBDA(int i, bool& result) {
                            if (view2(0,i) != i) {
                              result = false;
                              printf("Mismatch at i=%d, view2(0,i)=%d\n", i, view2(0,i));
                              }},
          Kokkos::LAnd<bool>(all_correct));
  EXPECT_TRUE(all_correct);
  }

TEST(DeviceTest, resizeGPUTabDOWN){

  TRUSTTab<int> tab(4, 6);

  tab(0,0)=0;
  tab(0,1)=1;
  tab(0,2)=2;
  tab(0,3)=3;
  tab(0,4)=4;
  tab(0,5)=5;

  auto view = tab.view_rw();
  tab.resize(2,3);
  auto view2 = tab.view_rw();

  EXPECT_EQ(tab(0,0), 0);
  EXPECT_EQ(tab(0,1), 1);
  EXPECT_EQ(tab(0,2), 2);
  EXPECT_EQ(tab(1,0), 3);
  EXPECT_EQ(tab(1,1), 4);
  EXPECT_EQ(tab(1,2), 5);

  bool all_correct = true;
  Kokkos::parallel_reduce("CheckValues", 6,
                          KOKKOS_LAMBDA(int i, bool& result) {
                            int k=(int)(i>=3);
                            int l=i%3;
                            if (view2(k,l) != i) {result = false;
                             printf("Mismatch, i=%d, view(%d,%d)=%d\n", i, k,l, view2(k,l));}},
          Kokkos::LAnd<bool>(all_correct));
  EXPECT_TRUE(all_correct);
}


TEST(DeviceTest, inject_array_GPU){

  TRUSTTab<int> tab(3, 3);

  tab(0,0)=0;
  tab(0,1)=1;
  tab(0,2)=2;
  tab(1,0)=3;
  tab(1,1)=4;
  tab(1,2)=5;
  tab(2,0)=6;
  tab(2,1)=7;
  tab(2,2)=8;


  TRUSTArray<int> injection(3);
  injection(0)=-4;
  injection(1)=-5;
  injection(2)=-6;
  
  auto view0=tab.view_rw();
  auto view1=injection.view_ro();

  tab.inject_array(injection, 3, 4, 0);

  EXPECT_EQ(tab(0,0),0);
  EXPECT_EQ(tab(0,1),1);
  EXPECT_EQ(tab(0,2),2);
  EXPECT_EQ(tab(1,0),3);
  EXPECT_EQ(tab(1,1),-4);
  EXPECT_EQ(tab(1,2),-5);
  EXPECT_EQ(tab(2,0),-6);
  EXPECT_EQ(tab(2,1),7);
  EXPECT_EQ(tab(2,2),8);
}

TEST(DeviceTest, append_line_GPU_dim1){

  TRUSTTab<int> tab(5, 1);

  tab(0,0)=0;
  tab(1,0)=1;
  tab(2,0)=2;
  tab(3,0)=3;
  tab(4,0)=4;

  auto view0=tab.view_rw();

  tab.append_line(5);

  EXPECT_EQ(tab(0,0), 0);
  EXPECT_EQ(tab(1,0), 1);
  EXPECT_EQ(tab(2,0), 2);
  EXPECT_EQ(tab(3,0), 3);
  EXPECT_EQ(tab(4,0), 4);
  EXPECT_EQ(tab(5,0), 5);

  auto view1=tab.view_rw();

  bool all_correct = true;
  Kokkos::parallel_reduce("CheckValues", 5,
                          KOKKOS_LAMBDA(int i, bool& result) {
                            if (view1(i,0) != i) {result = false;}},
          Kokkos::LAnd<bool>(all_correct));
  EXPECT_TRUE(all_correct);

}

TEST(DeviceTest, append_line_GPU_dim2){

  TRUSTTab<int> tab(5, 2);

  tab(0,0)=0;
  tab(1,0)=1;
  tab(2,0)=2;
  tab(3,0)=3;
  tab(4,0)=4;

  tab(0,1)=0;
  tab(1,1)=1;
  tab(2,1)=2;
  tab(3,1)=3;
  tab(4,1)=4;

  auto view0=tab.view_rw();

  tab.append_line(5, 5);

  EXPECT_EQ(tab(0,0), 0);
  EXPECT_EQ(tab(1,0), 1);
  EXPECT_EQ(tab(2,0), 2);
  EXPECT_EQ(tab(3,0), 3);
  EXPECT_EQ(tab(4,0), 4);
  EXPECT_EQ(tab(5,0), 5);

  EXPECT_EQ(tab(0,1), 0);
  EXPECT_EQ(tab(1,1), 1);
  EXPECT_EQ(tab(2,1), 2);
  EXPECT_EQ(tab(3,1), 3);
  EXPECT_EQ(tab(4,1), 4);
  EXPECT_EQ(tab(5,1), 5);

  auto view1=tab.view_rw();

  bool all_correct = true;
  Kokkos::parallel_reduce("CheckValues", 5,
                          KOKKOS_LAMBDA(int i, bool& result) {
                            if (view1(i,0) != i) {result = false;}},
          Kokkos::LAnd<bool>(all_correct));
  EXPECT_TRUE(all_correct);

  all_correct = true;
  Kokkos::parallel_reduce("CheckValues", 5,
                          KOKKOS_LAMBDA(int i, bool& result) {
                            if (view1(i,1) != i) {result = false;}},
          Kokkos::LAnd<bool>(all_correct));
  EXPECT_TRUE(all_correct);

}

TEST(DeviceTest, append_line_GPU_dim3) {
  TRUSTTab<int> tab(5, 3);

  // Initialize first 5 rows
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 3; j++) {
      tab(i, j) = i;
    }
  }

  auto view0 = tab.view_rw();

  // Append a new row with all elements set to 5
  tab.append_line(5, 5, 5);

  // Check all values directly
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 3; j++) {
      EXPECT_EQ(tab(i, j), i);
    }
  }
  
  // Check appended row
  EXPECT_EQ(tab(5, 0), 5);
  EXPECT_EQ(tab(5, 1), 5);
  EXPECT_EQ(tab(5, 2), 5);

  auto view1 = tab.view_rw();

  // Verify with parallel_reduce for each column
  for (int col = 0; col < 3; col++) {
    bool all_correct = true;
    Kokkos::parallel_reduce("CheckValues", 5,
                            KOKKOS_LAMBDA(int i, bool& result) {
                              if (view1(i, col) != i) {result = false;}
                            },
                            Kokkos::LAnd<bool>(all_correct));
    EXPECT_TRUE(all_correct);
  }
}

TEST(DeviceTest, append_line_GPU_dim4) {
  TRUSTTab<int> tab(5, 4);

  // Initialize first 5 rows
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 4; j++) {
      tab(i, j) = i;
    }
  }

  auto view0 = tab.view_rw();

  // Append a new row with all elements set to 5
  tab.append_line(5, 5, 5, 5);

  // Check all values directly
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(tab(i, j), i);
    }
  }
  
  // Check appended row
  EXPECT_EQ(tab(5, 0), 5);
  EXPECT_EQ(tab(5, 1), 5);
  EXPECT_EQ(tab(5, 2), 5);
  EXPECT_EQ(tab(5, 3), 5);

  auto view1 = tab.view_rw();

  // Verify with parallel_reduce for each column
  for (int col = 0; col < 4; col++) {
    bool all_correct = true;
    Kokkos::parallel_reduce("CheckValues", 5,
                            KOKKOS_LAMBDA(int i, bool& result) {
                              if (view1(i, col) != i) {result = false;}
                            },
                            Kokkos::LAnd<bool>(all_correct));
    EXPECT_TRUE(all_correct);
  }
}

TEST(DeviceTest, copy_ctor_tab_GPU) {
  TRUSTTab<int> tab(5, 4);

  // Initialize first 5 rows
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 4; j++) {
      tab(i, j) = i;
    }
  }

  auto view0 = tab.view_rw();

  TRUSTTab<int> tab2(tab);

  auto view2 = tab2.view_rw();

  // Verify with parallel_reduce for each column
  for (int col = 0; col < 4; col++) {
    bool all_correct = true;
    Kokkos::parallel_reduce("CheckValues", 5,
                            KOKKOS_LAMBDA(int i, bool& result) {
                              if (view2(i, col) != i) {result = false;}
                            },
                            Kokkos::LAnd<bool>(all_correct));
    EXPECT_TRUE(all_correct);
  }
}


#ifdef __NVCOMPILER
#pragma diag_default 177
#endif


#endif //TRUST_USE_GPU
