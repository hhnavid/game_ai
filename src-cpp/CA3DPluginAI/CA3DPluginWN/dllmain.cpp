//-------------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//    File      : dllmain.cpp
//    Project   : CodeArt Engine
//    Authors   : Ali Salmanizadegan, Dr. Seyed Navid Hosseini
//    Date      : 2025-07-13
//    Time      : 20:22:00
//    Copyright : (c) CODEART ENGINE Corporation. All rights reserved.
//-----------------------------------------------------------------------------
//-------------------------------------------------------------------------------

#include "targetver.h"
#include "stdafx.h"

#define WIN32_LEAN_AND_MEAN
#define EXP __declspec(dllexport)

// Windows Header Files
#include <typeinfo>
#include <windows.h>
#include <thread>

#include "./AI/AiTypes.h"

BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}

	return TRUE;
}

typedef void(*CallbackFunctionVoid)(void(*)());
typedef void(*CallbackFunctionBool)(void(*)(bool));
typedef void(*CallbackFunctionInt)(void(*)(int));
typedef void(*CallbackFunctionIntInt)(void(*)(int, int));
typedef void(*CallbackFunctionIntIntFloat)(void(*)(int, int, float *));
typedef void(*CallbackFunctionIntIntIntFloat)(void(*)(int, int, int &, float **));
typedef void(*CallbackFunctionIntIntFloatFloat)(void(*)(int, int, float *, float *));
typedef void(*CallbackFunctionIntIntFloatVFloatV)(void(*)(int, int, float &, float &));
typedef void(*CallbackFunctionIntIntIntFloatFloat)(void(*)(int, int, int&, float *, float **));
typedef void(*CallbackFunctionIntIntIntInt)(void(*)(int, int, int&, int&));
typedef void(*CallbackFunctionVecVecVecFloatIntBool)(void(*)(float*, float*, float*, float, int&, bool&));


typedef void(*CallbackFunctionVoidCaller)();
typedef void(*CallbackFunctionBoolCaller)(bool);
typedef void(*CallbackFunctionIntCaller)(int);
typedef void(*CallbackFunctionIntIntCaller)(int, int);
typedef void(*CallbackFunctionIntIntFloatCaller)(int, int, float *);
typedef void(*CallbackFunctionIntIntIntFloatCaller)(int, int, int &, float **);
typedef void(*CallbackFunctionIntIntFloatFloatCaller)(int, int, float *, float *);
typedef void(*CallbackFunctionIntIntFloatVFloatVCaller)(int, int, float &, float &);
typedef void(*CallbackFunctionIntIntIntFloatFloatCaller)(int, int, int&, float *, float **);
typedef void(*CallbackFunctionIntIntIntIntCaller)(int, int, int&, int&);
typedef void(*CallbackFunctionVecVecVecFloatIntBoolCaller)(float*, float*, float*, float, int&, bool&);

extern "C"
{
#pragma region Game

	CallbackFunctionBool Game_Play = NULL;
	EXP void SetGame_Play_Callback(CallbackFunctionBool f) { Game_Play = f; }

#pragma endregion

#pragma region Level

	CallbackFunctionInt Level_Reload = NULL;
	EXP void SetLevel_Reload_Callback(CallbackFunctionInt f) { Level_Reload = f; }

#pragma endregion

#pragma region Spline

	CallbackFunctionIntIntIntFloat Spline_GetPoints = NULL;
	EXP void SetSpline_GetPoints_Callback(CallbackFunctionIntIntIntFloat f) { Spline_GetPoints = f; }
	CallbackFunctionIntIntIntFloatFloat Spline_GetNearestPoints = NULL;
	EXP void SetSpline_GetNearestPoints_Callback(CallbackFunctionIntIntIntFloatFloat f) { Spline_GetNearestPoints = f; }
	CallbackFunctionIntIntFloat Spline_GetWidth = NULL;
	EXP void SetSpline_GetWidth_Callback(CallbackFunctionIntIntFloat f) { Spline_GetWidth = f; }
	CallbackFunctionIntIntFloatFloat Spline_GetWayPercent = NULL;
	EXP void SetSpline_GetWayPercent_Callback(CallbackFunctionIntIntFloatFloat f) { Spline_GetWayPercent = f; }

#pragma endregion

#pragma region Vehicle

	CallbackFunctionIntIntFloat Vehicle_GetPosition = NULL;
	EXP void SetVehicle_GetPosition_Callback(CallbackFunctionIntIntFloat f) { Vehicle_GetPosition = f; }
	CallbackFunctionIntIntFloat Vehicle_GetForwardVector = NULL;
	EXP void SetVehicle_GetForwardVector_Callback(CallbackFunctionIntIntFloat f) { Vehicle_GetForwardVector = f; }
	CallbackFunctionIntIntIntInt Vehicle_HasCollided = NULL;
	EXP void SetVehicle_HasCollided_Callback(CallbackFunctionIntIntIntInt f) { Vehicle_HasCollided = f; }
	CallbackFunctionIntInt Vehicle_Forward = NULL;
	EXP void SetVehicle_Forward_Callback(CallbackFunctionIntInt f) { Vehicle_Forward = f; }
	CallbackFunctionIntInt Vehicle_Backward = NULL;
	EXP void SetVehicle_Backward_Callback(CallbackFunctionIntInt f) { Vehicle_Backward = f; }
	CallbackFunctionIntInt Vehicle_Left = NULL;
	EXP void SetVehicle_Left_Callback(CallbackFunctionIntInt f) { Vehicle_Left = f; }
	CallbackFunctionIntInt Vehicle_Right = NULL;
	EXP void SetVehicle_Right_Callback(CallbackFunctionIntInt f) { Vehicle_Right = f; }
	CallbackFunctionIntInt Vehicle_ReleaseForwadBackward = NULL;
	EXP void SetVehicle_ReleaseForwadBackward_Callback(CallbackFunctionIntInt f) { Vehicle_ReleaseForwadBackward = f; }
	CallbackFunctionIntInt Vehicle_ReleaseLeftRight = NULL;
	EXP void SetVehicle_ReleaseLeftRight_Callback(CallbackFunctionIntInt f) { Vehicle_ReleaseLeftRight = f; }
	CallbackFunctionIntInt Vehicle_Handbrake = NULL;
	EXP void SetVehicle_Handbrake_Callback(CallbackFunctionIntInt f) { Vehicle_Handbrake = f; }
	CallbackFunctionIntInt Vehicle_ReleaseHandbrake = NULL;
	EXP void SetVehicle_ReleaseHandbrake_Callback(CallbackFunctionIntInt f) { Vehicle_ReleaseHandbrake = f; }
	CallbackFunctionIntIntFloatVFloatV Vehicle_GetSpeed = NULL;
	EXP void SetVehicle_GetSpeed_Callback(CallbackFunctionIntIntFloatVFloatV f) { Vehicle_GetSpeed = f; }
	CallbackFunctionIntIntFloat Vehicle_SetPosition = NULL;
	EXP void SetVehicle_SetPosition_Callback(CallbackFunctionIntIntFloat f) { Vehicle_SetPosition = f; }
	CallbackFunctionIntIntFloatFloat Vehicle_ForwardBackwardLeftRight = NULL;
	EXP void SetVehicle_ForwardBackwardLeftRight_Callback(CallbackFunctionIntIntFloatFloat f) { Vehicle_ForwardBackwardLeftRight = f; }

#pragma endregion

#pragma region Timer2D

	CallbackFunctionIntInt Timer2D_Reset = NULL;
	EXP void SetTimer2D_Reset_Callback(CallbackFunctionIntInt f) { Timer2D_Reset = f; }
	CallbackFunctionIntIntFloat Timer2D_GetTimerTime = NULL;
	EXP void SetTimer2D_GetTimerTime_Callback(CallbackFunctionIntIntFloat f) { Timer2D_GetTimerTime = f; }

#pragma endregion

#pragma region Math

	CallbackFunctionVecVecVecFloatIntBool Ray_Test = NULL;
	EXP void SetRay_Test_Callback(CallbackFunctionVecVecVecFloatIntBool f) { Ray_Test = f; }

#pragma endregion

#pragma region StaticMesh

	CallbackFunctionIntIntFloat StaticMesh_GetPosition = NULL;
	EXP void SetStaticMesh_GetPosition_Callback(CallbackFunctionIntIntFloat f) { StaticMesh_GetPosition = f; }
	CallbackFunctionIntIntFloat StaticMesh_SetPosition = NULL;
	EXP void SetStaticMesh_SetPosition_Callback(CallbackFunctionIntIntFloat f) { StaticMesh_SetPosition = f; }

#pragma endregion

#pragma region Init
	void SecondThread();

	EXP void Init()
	{
		static bool init = false;

		if (!init)
		{
			init = true;
			std::thread StreamThread(SecondThread);
			StreamThread.detach();
		}
	}

#pragma endregion

#pragma region Development

	void SecondThread()
	{
		while (true)
		{
			/* >>>>>>>>>> ADD YOUR CODE HERE <<<<<<<<<< */
			float position[3] = { 0, 0, 0 };

			((CallbackFunctionIntIntFloatCaller)StaticMesh_GetPosition)(0, 1, position);
			position[0] += 0.1f; // add to x
			position[1] += 0.2f;
			((CallbackFunctionIntIntFloatCaller)StaticMesh_SetPosition)(0, 1, position);

			Sleep(100); // 300 ms period time to get, check and change position of objects
		}
	}

#pragma endregion
}