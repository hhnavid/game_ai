//-------------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//    File      : Matrix.cpp
//    Project   : FruitBall Android
//    Editor    : Ali Salmanizadegan
//    Date      : 1394\--\--
//    Edit		: 1397\02\08
//    Time      : 18:05:00
//    Copyright : (c) Codeart3D Corporation. All rights reserved.
//-----------------------------------------------------------------------------
//-------------------------------------------------------------------------------
#include "stdafx.h"
#include "Matrix.h"

void Matrix2Rotation(MATRIX2 *m, float t)
{
	float cost = cos(t);
	float sint = sin(t);

	m->m[0].x = cost;
	m->m[0].y = -sint;

	m->m[1].x = sint;
	m->m[1].y = cost;
}

void Vector2MultiplyMatrix2(VECTOR2 * v, MATRIX2 * m)
{
	VECTOR2 u = *v;

	v->x = (u.x * m->m[0].x) + (u.y * m->m[1].x);
	v->y = (u.x * m->m[0].y) + (u.y * m->m[1].y);
}

void Vector3MultiplyMatrix3(VECTOR3 * v, MATRIX3 * m)
{
	VECTOR3 u = *v;
	v->x = (u.x * m->m[0].x) + (u.y * m->m[1].x) + (u.z * m->m[2].x);
	v->y = (u.x * m->m[0].y) + (u.y * m->m[1].y) + (u.z * m->m[2].y);
	v->z = (u.x * m->m[0].z) + (u.y * m->m[1].z) + (u.z * m->m[2].z);
}

void Vector3MultiplyMatrix3(VECTOR3 *out, VECTOR3 *v, MATRIX3 *m)
{
	out->x = (v->x * m->m[0].x) + (v->y * m->m[1].x) + (v->z * m->m[2].x);
	out->y = (v->x * m->m[0].y) + (v->y * m->m[1].y) + (v->z * m->m[2].y);
	out->z = (v->x * m->m[0].z) + (v->y * m->m[1].z) + (v->z * m->m[2].z);
}

void Vector3MultiplyMatrix4(VECTOR3 *out, VECTOR3 v, MATRIX4 *m)
{
	out->x = (v.x * m->m[0].x) + (v.y * m->m[1].x) + (v.z * m->m[2].x);
	out->y = (v.x * m->m[0].y) + (v.y * m->m[1].y) + (v.z * m->m[2].y);
	out->z = (v.x * m->m[0].z) + (v.y * m->m[1].z) + (v.z * m->m[2].z);
}

void Vector3MultiplyMatrix4(VECTOR3 *out, VECTOR3 *v, MATRIX4 *m)
{
	out->x = (v->x * m->m[0].x) + (v->y * m->m[1].x) + (v->z * m->m[2].x);
	out->y = (v->x * m->m[0].y) + (v->y * m->m[1].y) + (v->z * m->m[2].y);
	out->z = (v->x * m->m[0].z) + (v->y * m->m[1].z) + (v->z * m->m[2].z);
}

void Vector3TransformCoord(VECTOR3 *pout, const VECTOR3 *pv, const MATRIX4 *pm)
{
	VECTOR3 out;
	float norm = pm->m[0].z * pv->x + pm->m[1].z * pv->y + pm->m[2].z * pv->z + pm->m[3].z;

	/*out.x = (pm->m[0].x * pv->x + pm->m[1].x * pv->y + pm->m[2].x * pv->z + pm->m[3].x) / norm;
	out.y = (pm->m[0].y * pv->x + pm->m[1].y * pv->y + pm->m[2].y * pv->z + pm->m[3].y) / norm;
	out.z = (pm->m[0].z * pv->x + pm->m[1].z * pv->y + pm->m[2].z * pv->z + pm->m[3].z) / norm;*/

	out.x = ((pv->x * pm->m[0].x) + (pv->y * pm->m[1].x) + (pv->z * pm->m[2].x) + pm->m[3].x) / norm;
	out.y = ((pv->x * pm->m[0].y) + (pv->y * pm->m[1].y) + (pv->z * pm->m[2].y) + pm->m[3].y) / norm;
	out.z = ((pv->x * pm->m[0].z) + (pv->y * pm->m[1].z) + (pv->z * pm->m[2].z) + pm->m[3].z) / norm;

	*pout = out;
}

void Vector3TransformNormal(VECTOR3 * pout, const VECTOR3 * pv, const MATRIX4 * pm)
{
	VECTOR3 out;
	float norm = pm->m[0].z * pv->x + pm->m[1].z * pv->y + pm->m[2].z * pv->z + pm->m[3].z;

	out.x = (pm->m[0].x * pv->x + pm->m[1].x * pv->y + pm->m[2].x * pv->z + pm->m[3].x) / norm;
	out.y = (pm->m[0].y * pv->x + pm->m[1].y * pv->y + pm->m[2].y * pv->z + pm->m[3].y) / norm;
	out.z = (pm->m[0].z * pv->x + pm->m[1].z * pv->y + pm->m[2].z * pv->z + pm->m[3].z) / norm;

	*pout = out;
}

void Vector4MultiplyMatrix4(VECTOR4 * out, VECTOR4 v, MATRIX4 * m)
{
	out->x = (v.x * m->m[0].x) + (v.y * m->m[1].x) + (v.z * m->m[2].x) + (v.w * m->m[3].x);
	out->y = (v.x * m->m[0].y) + (v.y * m->m[1].y) + (v.z * m->m[2].y) + (v.w * m->m[3].y);
	out->z = (v.x * m->m[0].z) + (v.y * m->m[1].z) + (v.z * m->m[2].z) + (v.w * m->m[3].z);
	out->w = (v.x * m->m[0].w) + (v.y * m->m[1].w) + (v.z * m->m[2].w) + (v.w * m->m[3].w);
}

void Vector4MultiplyMatrix4(VECTOR4 *out, VECTOR4 *v, MATRIX4 *m)
{
	out->x = (v->x * m->m[0].x) + (v->y * m->m[1].x) + (v->z * m->m[2].x) + (v->w * m->m[3].x);
	out->y = (v->x * m->m[0].y) + (v->y * m->m[1].y) + (v->z * m->m[2].y) + (v->w * m->m[3].y);
	out->z = (v->x * m->m[0].z) + (v->y * m->m[1].z) + (v->z * m->m[2].z) + (v->w * m->m[3].z);
	out->w = (v->x * m->m[0].w) + (v->y * m->m[1].w) + (v->z * m->m[2].w) + (v->w * m->m[3].w);
}

void Matrix3Zero(MATRIX3 *m)
{
	memset(&m[0], 0, sizeof(MATRIX3));
}

void Matrix3Identity(MATRIX3 *m)
{
	m->m[0].y = 0.0f;
	m->m[0].z = 0.0f;

	m->m[1].x = 0.0f;
	m->m[1].z = 0.0f;

	m->m[2].x = 0.0f;
	m->m[2].y = 0.0f;

	m->m[0].x = 1.0f;
	m->m[1].y = 1.0f;
	m->m[2].z = 1.0f;
}

void Matrix3Quaternion(MATRIX3 * out, VECTOR4 q)
{
	float xx = q.x * q.x;
	float yy = q.y * q.y;
	float zz = q.z * q.z;
	float xy = q.x * q.y;
	float xz = q.x * q.z;
	float yz = q.y * q.z;
	float wx = q.w * q.x;
	float wy = q.w * q.y;
	float wz = q.w * q.z;

	out->m[0].x = 1.0f - 2.0f * yy - 2.0f * zz;
	out->m[0].y = 2.0f * (xy + wz);
	out->m[0].z = 2.0f * (xz - wy);

	out->m[1].x = 2.0f * (xy - wz);
	out->m[1].y = 1.0f - 2.0f * xx - 2.0f * zz;
	out->m[1].z = 2.0f * (yz + wx);

	out->m[2].x = 2.0f * (xz + wy);
	out->m[2].y = 2.0f * (yz - wx);
	out->m[2].z = 1.0f - 2.0f * xx - 2.0f * yy;
}

MATRIX3 Matrix3Translation(VECTOR2 &trans)
{
	MATRIX3 res;

	res.m[0].x = 1.0f;
	res.m[0].y = 0.0f;
	res.m[0].z = 0.0f;

	res.m[1].x = 0.0f;
	res.m[1].y = 1.0f;
	res.m[1].z = 0.0f;

	res.m[2].x = trans.x;
	res.m[2].y = trans.y;
	res.m[2].z = 1.0f;

	return res;
}

MATRIX3 Matrix3TranslationInverse(VECTOR2 &trans)
{
	MATRIX3 res;

	float d = 0;
	float m[3][3];
	/*m[0] = (float*)&(res.m[0].x);
	m[1] = (float*)&(res.m[1].x);
	m[2] = (float*)&(res.m[2].x);*/
	
	m[0][0] = 1.0f;
	m[0][1] = 0.0f;
	m[0][2] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = 1.0f;
	m[1][2] = 0.0f;

	m[2][0] = trans.x;
	m[2][1] = trans.y;
	m[2][2] = 1.0f;

	for (int i = 0; i < 3; i++)
		d = d + (m[0][i] * (m[1][(i + 1) % 3] * m[2][(i + 2) % 3] - m[1][(i + 2) % 3] * m[2][(i + 1) % 3]));

	if (d > 0)
	{
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++)
				m[i][j] = ((m[(j + 1) % 3][(i + 1) % 3] * m[(j + 2) % 3][(i + 2) % 3]) - (m[(j + 1) % 3][(i + 2) % 3] * m[(j + 2) % 3][(i + 1) % 3])) / d;
		}
	}

	res.m[0] = VECTOR3(m[0][0], m[0][1], m[0][2]);
	res.m[1] = VECTOR3(m[1][0], m[1][1], m[1][2]);
	res.m[2] = VECTOR3(m[2][0], m[2][1], m[2][2]);

	return res;
}

MATRIX3 Matrix3Rotation(float t)
{
	MATRIX3 res;

	t *= DEG_TO_RAD;
	float cost = cosf(t);
	float sint = sinf(t);

	res.m[0].x = cost;
	res.m[0].y = -sint;
	res.m[0].z = 0.0f;

	res.m[1].x = sint;
	res.m[1].y = cost;
	res.m[1].z = 0.0f;

	res.m[2].x = 0.0f;
	res.m[2].y = 0.0f;
	res.m[2].z = 1.0f;

	return res;
}

void Matrix4Identity(MATRIX4 *m)
{
	m->m[0].y = 0.0f;
	m->m[0].z = 0.0f;
	m->m[0].w = 0.0f;

	m->m[1].x = 0.0f;
	m->m[1].z = 0.0f;
	m->m[1].w = 0.0f;

	m->m[2].x = 0.0f;
	m->m[2].y = 0.0f;
	m->m[2].w = 0.0f;

	m->m[3].x = 0.0f;
	m->m[3].y = 0.0f;
	m->m[3].z = 0.0f;

	m->m[0].x = 1.0f;
	m->m[1].y = 1.0f;
	m->m[2].z = 1.0f;
	m->m[3].w = 1.0f;
}

void Matrix4Translate(MATRIX4 *out, MATRIX4 *m, VECTOR3 *v)
{
	out->m[3].x = m->m[0].x * v->x + m->m[1].x * v->y + m->m[2].x * v->z + m->m[3].x;
	out->m[3].y = m->m[0].y * v->x + m->m[1].y * v->y + m->m[2].y * v->z + m->m[3].y;
	out->m[3].z = m->m[0].z * v->x + m->m[1].z * v->y + m->m[2].z * v->z + m->m[3].z;
	out->m[3].w = m->m[0].w * v->x + m->m[1].w * v->y + m->m[2].w * v->z + m->m[3].w;
}

void Matrix4RotateFast(MATRIX4 *m, VECTOR4 *v)
{
	float value = v->w * DEG_TO_RAD;
	float s = sinf(value);
	float c = cosf(value);

	MATRIX4 mat;
	Matrix4Identity(&mat);

	if (!v->x)
	{
		if (!v->y)
		{
			if (v->z)
			{
				mat.m[0].x = c;
				mat.m[1].y = c;

				if (v->z < 0.0f)
				{
					mat.m[1].x = s;
					mat.m[0].y = -s;
				}
				else
				{
					mat.m[1].x = -s;
					mat.m[0].y = s;
				}
			}
		}
		else if (!v->z)
		{
			mat.m[0].x = c;
			mat.m[2].z = c;

			if (v->y < 0.0f)
			{
				mat.m[2].x = -s;
				mat.m[0].z = s;
			}
			else
			{
				mat.m[2].x = s;
				mat.m[0].z = -s;
			}
		}
	}
	else if (!v->y)
	{
		if (!v->z)
		{
			mat.m[1].y = c;
			mat.m[2].z = c;

			if (v->x < 0.0f)
			{
				mat.m[2].y = s;
				mat.m[1].z = -s;
			}
			else
			{
				mat.m[2].y = -s;
				mat.m[1].z = s;
			}
		}
	}

	Matrix4MultiplyMatrix4(m, m, &mat);
}

/*void Matrix4Rotate(MATRIX4 *out, MATRIX4 *m, VECTOR4 *v)
{
	float value = v->w * DEG_TO_RAD;
	float s = sinf(value);
	float c = cosf(value);
	float xx, yy, zz, xy, yz, zx, xs, ys, zs, c1;

	MATRIX4 mat;
	Matrix4Identity(&mat);
	VECTOR3 t = VECTOR3(v->x, v->y, v->z);

	if (!v->w || !Vector3Normalize(&t, &t))
		return;

	xx = t.x * t.x;
	yy = t.y * t.y;
	zz = t.z * t.z;
	xy = t.x * t.y;
	yz = t.y * t.z;
	zx = t.z * t.x;
	xs = t.x * s;
	ys = t.y * s;
	zs = t.z * s;
	c1 = 1.0f - c;

	mat.m[0].x = (c1 * xx) + c;
	mat.m[1].x = (c1 * xy) - zs;
	mat.m[2].x = (c1 * zx) + ys;

	mat.m[0].y = (c1 * xy) + zs;
	mat.m[1].y = (c1 * yy) + c;
	mat.m[2].y = (c1 * yz) - xs;

	mat.m[0].z = (c1 * zx) - ys;
	mat.m[1].z = (c1 * yz) + xs;
	mat.m[2].z = (c1 * zz) + c;

	Matrix4MultiplyMatrix4(out, m, &mat);
}*/

void Matrix4Scale(MATRIX4 *out, MATRIX4 *m, VECTOR3 *v)
{
	out->m[0].x = m->m[0].x * v->x;
	out->m[0].y = m->m[0].y * v->x;
	out->m[0].z = m->m[0].z * v->x;
	out->m[0].w = m->m[0].w * v->x;

	out->m[1].x = m->m[1].x * v->y;
	out->m[1].y = m->m[1].y * v->y;
	out->m[1].z = m->m[1].z * v->y;
	out->m[1].w = m->m[1].w * v->y;

	out->m[2].x = m->m[2].x * v->z;
	out->m[2].y = m->m[2].y * v->z;
	out->m[2].z = m->m[2].z * v->z;
	out->m[2].w = m->m[2].w * v->z;
}

bool Matrix4Invert(MATRIX4 *m)
{
	MATRIX4 mat;
	float d = m->m[0].x * m->m[0].x + m->m[1].x * m->m[1].x + m->m[2].x * m->m[2].x;

	if (!d)
		return false;

	d = 1.0f / d;

	mat.m[0].x = d * m->m[0].x;
	mat.m[0].y = d * m->m[1].x;
	mat.m[0].z = d * m->m[2].x;

	mat.m[1].x = d * m->m[0].y;
	mat.m[1].y = d * m->m[1].y;
	mat.m[1].z = d * m->m[2].y;

	mat.m[2].x = d * m->m[0].z;
	mat.m[2].y = d * m->m[1].z;
	mat.m[2].z = d * m->m[2].z;

	mat.m[3].x = -(mat.m[0].x * m->m[3].x + mat.m[1].x * m->m[3].y + mat.m[2].x * m->m[3].z);
	mat.m[3].y = -(mat.m[0].y * m->m[3].x + mat.m[1].y * m->m[3].y + mat.m[2].y * m->m[3].z);
	mat.m[3].z = -(mat.m[0].z * m->m[3].x + mat.m[1].z * m->m[3].y + mat.m[2].z * m->m[3].z);

	mat.m[0].w = 0.0f;
	mat.m[1].w = 0.0f;
	mat.m[2].w = 0.0f;
	mat.m[3].w = 1.0f;

	*m = mat;

	return true;
}

bool Matrix4InvertFull(MATRIX4 *m)
{
	MATRIX4 inv;

	float d;

	inv.m[0].x = m->m[1].y * m->m[2].z * m->m[3].w -
		m->m[1].y * m->m[2].w * m->m[3].z -
		m->m[2].y * m->m[1].z * m->m[3].w +
		m->m[2].y * m->m[1].w * m->m[3].z +
		m->m[3].y * m->m[1].z * m->m[2].w -
		m->m[3].y * m->m[1].w * m->m[2].z;

	inv.m[1].x = -m->m[1].x * m->m[2].z * m->m[3].w +
		m->m[1].x * m->m[2].w * m->m[3].z +
		m->m[2].x * m->m[1].z * m->m[3].w -
		m->m[2].x * m->m[1].w * m->m[3].z -
		m->m[3].x * m->m[1].z * m->m[2].w +
		m->m[3].x * m->m[1].w * m->m[2].z;

	inv.m[2].x = m->m[1].x * m->m[2].y * m->m[3].w -
		m->m[1].x * m->m[2].w * m->m[3].y -
		m->m[2].x * m->m[1].y * m->m[3].w +
		m->m[2].x * m->m[1].w * m->m[3].y +
		m->m[3].x * m->m[1].y * m->m[2].w -
		m->m[3].x * m->m[1].w * m->m[2].y;

	inv.m[3].x = -m->m[1].x * m->m[2].y * m->m[3].z +
		m->m[1].x * m->m[2].z * m->m[3].y +
		m->m[2].x * m->m[1].y * m->m[3].z -
		m->m[2].x * m->m[1].z * m->m[3].y -
		m->m[3].x * m->m[1].y * m->m[2].z +
		m->m[3].x * m->m[1].z * m->m[2].y;

	inv.m[0].y = -m->m[0].y * m->m[2].z * m->m[3].w +
		m->m[0].y * m->m[2].w * m->m[3].z +
		m->m[2].y * m->m[0].z * m->m[3].w -
		m->m[2].y * m->m[0].w * m->m[3].z -
		m->m[3].y * m->m[0].z * m->m[2].w +
		m->m[3].y * m->m[0].w * m->m[2].z;

	inv.m[1].y = m->m[0].x * m->m[2].z * m->m[3].w -
		m->m[0].x * m->m[2].w * m->m[3].z -
		m->m[2].x * m->m[0].z * m->m[3].w +
		m->m[2].x * m->m[0].w * m->m[3].z +
		m->m[3].x * m->m[0].z * m->m[2].w -
		m->m[3].x * m->m[0].w * m->m[2].z;

	inv.m[2].y = -m->m[0].x * m->m[2].y * m->m[3].w +
		m->m[0].x * m->m[2].w * m->m[3].y +
		m->m[2].x * m->m[0].y * m->m[3].w -
		m->m[2].x * m->m[0].w * m->m[3].y -
		m->m[3].x * m->m[0].y * m->m[2].w +
		m->m[3].x * m->m[0].w * m->m[2].y;

	inv.m[3].y = m->m[0].x * m->m[2].y * m->m[3].z -
		m->m[0].x * m->m[2].z * m->m[3].y -
		m->m[2].x * m->m[0].y * m->m[3].z +
		m->m[2].x * m->m[0].z * m->m[3].y +
		m->m[3].x * m->m[0].y * m->m[2].z -
		m->m[3].x * m->m[0].z * m->m[2].y;

	inv.m[0].z = m->m[0].y * m->m[1].z * m->m[3].w -
		m->m[0].y * m->m[1].w * m->m[3].z -
		m->m[1].y * m->m[0].z * m->m[3].w +
		m->m[1].y * m->m[0].w * m->m[3].z +
		m->m[3].y * m->m[0].z * m->m[1].w -
		m->m[3].y * m->m[0].w * m->m[1].z;

	inv.m[1].z = -m->m[0].x * m->m[1].z * m->m[3].w +
		m->m[0].x * m->m[1].w * m->m[3].z +
		m->m[1].x * m->m[0].z * m->m[3].w -
		m->m[1].x * m->m[0].w * m->m[3].z -
		m->m[3].x * m->m[0].z * m->m[1].w +
		m->m[3].x * m->m[0].w * m->m[1].z;

	inv.m[2].z = m->m[0].x * m->m[1].y * m->m[3].w -
		m->m[0].x * m->m[1].w * m->m[3].y -
		m->m[1].x * m->m[0].y * m->m[3].w +
		m->m[1].x * m->m[0].w * m->m[3].y +
		m->m[3].x * m->m[0].y * m->m[1].w -
		m->m[3].x * m->m[0].w * m->m[1].y;

	inv.m[3].z = -m->m[0].x * m->m[1].y * m->m[3].z +
		m->m[0].x * m->m[1].z * m->m[3].y +
		m->m[1].x * m->m[0].y * m->m[3].z -
		m->m[1].x * m->m[0].z * m->m[3].y -
		m->m[3].x * m->m[0].y * m->m[1].z +
		m->m[3].x * m->m[0].z * m->m[1].y;

	inv.m[0].w = -m->m[0].y * m->m[1].z * m->m[2].w +
		m->m[0].y * m->m[1].w * m->m[2].z +
		m->m[1].y * m->m[0].z * m->m[2].w -
		m->m[1].y * m->m[0].w * m->m[2].z -
		m->m[2].y * m->m[0].z * m->m[1].w +
		m->m[2].y * m->m[0].w * m->m[1].z;

	inv.m[1].w = m->m[0].x * m->m[1].z * m->m[2].w -
		m->m[0].x * m->m[1].w * m->m[2].z -
		m->m[1].x * m->m[0].z * m->m[2].w +
		m->m[1].x * m->m[0].w * m->m[2].z +
		m->m[2].x * m->m[0].z * m->m[1].w -
		m->m[2].x * m->m[0].w * m->m[1].z;

	inv.m[2].w = -m->m[0].x * m->m[1].y * m->m[2].w +
		m->m[0].x * m->m[1].w * m->m[2].y +
		m->m[1].x * m->m[0].y * m->m[2].w -
		m->m[1].x * m->m[0].w * m->m[2].y -
		m->m[2].x * m->m[0].y * m->m[1].w +
		m->m[2].x * m->m[0].w * m->m[1].y;

	inv.m[3].w = m->m[0].x * m->m[1].y * m->m[2].z -
		m->m[0].x * m->m[1].z * m->m[2].y -
		m->m[1].x * m->m[0].y * m->m[2].z +
		m->m[1].x * m->m[0].z * m->m[2].y +
		m->m[2].x * m->m[0].y * m->m[1].z -
		m->m[2].x * m->m[0].z * m->m[1].y;

	d = m->m[0].x * inv.m[0].x +
		m->m[0].y * inv.m[1].x +
		m->m[0].z * inv.m[2].x +
		m->m[0].w * inv.m[3].x;

	if (!d)
		return false;

	d = 1.0f / d;

	inv.m[0] *= d;
	inv.m[1] *= d;
	inv.m[2] *= d;
	inv.m[3] *= d;

	*m = inv;

	return true;
}

bool Matrix4Invert(MATRIX4 & inv, MATRIX4 & world)
{
	float d = world.m[0].x * world.m[0].x + world.m[1].x * world.m[1].x + world.m[2].x * world.m[2].x;

	if (!d)
		return false;

	d = 1.0f / d;

	inv.m[0].x = d * world.m[0].x;
	inv.m[0].y = d * world.m[1].x;
	inv.m[0].z = d * world.m[2].x;

	inv.m[1].x = d * world.m[0].y;
	inv.m[1].y = d * world.m[1].y;
	inv.m[1].z = d * world.m[2].y;

	inv.m[2].x = d * world.m[0].z;
	inv.m[2].y = d * world.m[1].z;
	inv.m[2].z = d * world.m[2].z;

	inv.m[3].x = -(inv.m[0].x * world.m[3].x + inv.m[1].x * world.m[3].y + inv.m[2].x * world.m[3].z);
	inv.m[3].y = -(inv.m[0].y * world.m[3].x + inv.m[1].y * world.m[3].y + inv.m[2].y * world.m[3].z);
	inv.m[3].z = -(inv.m[0].z * world.m[3].x + inv.m[1].z * world.m[3].y + inv.m[2].z * world.m[3].z);

	inv.m[0].w = 0.0f;
	inv.m[1].w = 0.0f;
	inv.m[2].w = 0.0f;
	inv.m[3].w = 1.0f;

	return true;
}

void Matrix4Transpose(MATRIX4 *m)
{
	float t;

	t = m->m[0].y;
	m->m[0].y = m->m[1].x;
	m->m[1].x = t;

	t = m->m[0].z;
	m->m[0].z = m->m[2].x;
	m->m[2].x = t;

	t = m->m[0].w;
	m->m[0].w = m->m[3].x;
	m->m[3].x = t;

	t = m->m[1].z;
	m->m[1].z = m->m[2].y;
	m->m[2].y = t;

	t = m->m[1].w;
	m->m[1].w = m->m[3].y;
	m->m[3].y = t;

	t = m->m[2].w;
	m->m[2].w = m->m[3].z;
	m->m[3].z = t;
}

void Matrix4Zoom(MATRIX4 *out, float zoom, float x, float y)
{
	out->m[0].x = zoom;
	out->m[1].x = 0.0f;
	out->m[2].x = 0.0f;
	out->m[3].x = x;

	out->m[0].y = 0.0f;
	out->m[1].y = zoom;
	out->m[2].y = 0.0f;
	out->m[3].y = y;

	out->m[0].z = 0.0f;
	out->m[1].z = 0.0f;
	out->m[2].z = 1.0f;
	out->m[3].z = 0.0f;

	out->m[0].w = 0.0f;
	out->m[1].w = 0.0f;
	out->m[2].w = 0.0f;
	out->m[3].w = 1.0f;
}

void Matrix4Ortho(MATRIX4 *out, float left, float right, float bottom, float top, float clip_start, float clip_end)
{
	out->m[0].x = 2.0f / (right - left);
	out->m[1].x = 0.0f;
	out->m[2].x = 0.0f;
	out->m[3].x = -(right + left) / (right - left);

	out->m[0].y = 0.0f;
	out->m[1].y = 2.0f / (top - bottom);
	out->m[2].y = 0.0f;
	out->m[3].y = -(top + bottom) / (top - bottom);

	out->m[0].z = 0.0f;
	out->m[1].z = 0.0f;
	out->m[2].z = -2.0f / (clip_end - clip_start);
	out->m[3].z = -(clip_end + clip_start) / (clip_end - clip_start);

	out->m[0].w = 0.0f;
	out->m[1].w = 0.0f;
	out->m[2].w = 0.0f;
	out->m[3].w = 1.0f;
}

void Matrix4Quaternion(MATRIX4 * out, VECTOR4 q)
{
	out->m[0].x = q.w;
	out->m[0].y = q.z;
	out->m[0].z = -q.y;
	out->m[0].w = q.x;

	out->m[1].x = -q.z;
	out->m[1].y = q.w;
	out->m[1].z = q.x;
	out->m[1].w = q.y;

	out->m[2].x = q.y;
	out->m[2].y = -q.x;
	out->m[2].z = q.w;
	out->m[2].w = q.z;

	out->m[3].x = -q.x;
	out->m[3].y = -q.y;
	out->m[3].z = -q.z;
	out->m[3].w = q.w;
}

MATRIX4 Matrix4RotationX(float t)
{
	MATRIX4 res;

	t *= DEG_TO_RAD;
	float cost = cosf(t);
	float sint = sinf(t);

	res.m[0].x = 1.0f;
	res.m[0].y = 0.0f;
	res.m[0].z = 0.0f;
	res.m[0].w = 0.0f;

	res.m[1].x = 0.0f;
	res.m[1].y = cost;
	res.m[1].z = sint;
	res.m[1].w = 0.0f;

	res.m[2].x = 0.0f;
	res.m[2].y = -sint;
	res.m[2].z = cost;
	res.m[2].w = 0.0f;

	res.m[3].x = 0.0f;
	res.m[3].y = 0.0f;
	res.m[3].z = 0.0f;
	res.m[3].w = 1.0f;

	return res;
}

MATRIX4 Matrix4RotationY(float t)
{
	MATRIX4 res;

	t *= DEG_TO_RAD;
	float cost = cosf(t);
	float sint = sinf(t);

	res.m[0].x = cost;
	res.m[0].y = 0.0f;
	res.m[0].z = sint;
	res.m[0].w = 0.0f;

	res.m[1].x = 0.0f;
	res.m[1].y = 1.0f;
	res.m[1].z = 0.0f;
	res.m[1].w = 0.0f;

	res.m[2].x = -sint;
	res.m[2].y = 0.0f;
	res.m[2].z = cost;
	res.m[2].w = 0.0f;

	res.m[3].x = 0.0f;
	res.m[3].y = 0.0f;
	res.m[3].z = 0.0f;
	res.m[3].w = 1.0f;

	return res;
}

MATRIX4 Matrix4RotationZ(float t)
{
	MATRIX4 res;

	t *= DEG_TO_RAD;
	float cost = cosf(t);
	float sint = sinf(t);

	res.m[0].x = cost;
	res.m[0].y = sint;
	res.m[0].z = 0.0f;
	res.m[0].w = 0.0f;

	res.m[1].x = -sint;
	res.m[1].y = cost;
	res.m[1].z = 0.0f;
	res.m[1].w = 0.0f;

	res.m[2].x = 0.0f;
	res.m[2].y = 0.0f;
	res.m[2].z = 1.0f;
	res.m[2].w = 0.0f;

	res.m[3].x = 0.0f;
	res.m[3].y = 0.0f;
	res.m[3].z = 0.0f;
	res.m[3].w = 1.0f;

	return res;
}

MATRIX4 Matrix4Rotation(float Yaw, float Pitch, float Roll)
{
	return Matrix4RotationZ(Pitch) * Matrix4RotationY(Yaw);// *Matrix4RotationZ(Roll);
}

void Matrix4CopyMatrix3(MATRIX4 *out, MATRIX3 *m)
{
	memcpy(&out->m[0], &m->m[0], sizeof(VECTOR3));
	memcpy(&out->m[1], &m->m[1], sizeof(VECTOR3));
	memcpy(&out->m[2], &m->m[2], sizeof(VECTOR3));
}

void Matrix4MultiplyMatrix3(MATRIX4 *out, MATRIX4 *m1, MATRIX3 *m2)
{
	MATRIX3 mat;

	mat.m[0].x = m1->m[0].x * m2->m[0].x + m1->m[1].x * m2->m[0].y + m1->m[2].x * m2->m[0].z;
	mat.m[0].y = m1->m[0].y * m2->m[0].x + m1->m[1].y * m2->m[0].y + m1->m[2].y * m2->m[0].z;
	mat.m[0].z = m1->m[0].z * m2->m[0].x + m1->m[1].z * m2->m[0].y + m1->m[2].z * m2->m[0].z;

	mat.m[1].x = m1->m[0].x * m2->m[1].x + m1->m[1].x * m2->m[1].y + m1->m[2].x * m2->m[1].z;
	mat.m[1].y = m1->m[0].y * m2->m[1].x + m1->m[1].y * m2->m[1].y + m1->m[2].y * m2->m[1].z;
	mat.m[1].z = m1->m[0].z * m2->m[1].x + m1->m[1].z * m2->m[1].y + m1->m[2].z * m2->m[1].z;

	mat.m[2].x = m1->m[0].x * m2->m[2].x + m1->m[1].x * m2->m[2].y + m1->m[2].x * m2->m[2].z;
	mat.m[2].y = m1->m[0].y * m2->m[2].x + m1->m[1].y * m2->m[2].y + m1->m[2].y * m2->m[2].z;
	mat.m[2].z = m1->m[0].z * m2->m[2].x + m1->m[1].z * m2->m[2].y + m1->m[2].z * m2->m[2].z;

	Matrix4CopyMatrix3(out, &mat);
}

void Matrix4MultiplyMatrix4(MATRIX4 *out, MATRIX4 *m1, MATRIX4 *m2)
{
	MATRIX4 mat;

	mat.m[0].x = m1->m[0].x * m2->m[0].x + m1->m[1].x * m2->m[0].y + m1->m[2].x * m2->m[0].z + m1->m[3].x * m2->m[0].w;
	mat.m[0].y = m1->m[0].y * m2->m[0].x + m1->m[1].y * m2->m[0].y + m1->m[2].y * m2->m[0].z + m1->m[3].y * m2->m[0].w;
	mat.m[0].z = m1->m[0].z * m2->m[0].x + m1->m[1].z * m2->m[0].y + m1->m[2].z * m2->m[0].z + m1->m[3].z * m2->m[0].w;
	mat.m[0].w = m1->m[0].w * m2->m[0].x + m1->m[1].w * m2->m[0].y + m1->m[2].w * m2->m[0].z + m1->m[3].w * m2->m[0].w;

	mat.m[1].x = m1->m[0].x * m2->m[1].x + m1->m[1].x * m2->m[1].y + m1->m[2].x * m2->m[1].z + m1->m[3].x * m2->m[1].w;
	mat.m[1].y = m1->m[0].y * m2->m[1].x + m1->m[1].y * m2->m[1].y + m1->m[2].y * m2->m[1].z + m1->m[3].y * m2->m[1].w;
	mat.m[1].z = m1->m[0].z * m2->m[1].x + m1->m[1].z * m2->m[1].y + m1->m[2].z * m2->m[1].z + m1->m[3].z * m2->m[1].w;
	mat.m[1].w = m1->m[0].w * m2->m[1].x + m1->m[1].w * m2->m[1].y + m1->m[2].w * m2->m[1].z + m1->m[3].w * m2->m[1].w;

	mat.m[2].x = m1->m[0].x * m2->m[2].x + m1->m[1].x * m2->m[2].y + m1->m[2].x * m2->m[2].z + m1->m[3].x * m2->m[2].w;
	mat.m[2].y = m1->m[0].y * m2->m[2].x + m1->m[1].y * m2->m[2].y + m1->m[2].y * m2->m[2].z + m1->m[3].y * m2->m[2].w;
	mat.m[2].z = m1->m[0].z * m2->m[2].x + m1->m[1].z * m2->m[2].y + m1->m[2].z * m2->m[2].z + m1->m[3].z * m2->m[2].w;
	mat.m[2].w = m1->m[0].w * m2->m[2].x + m1->m[1].w * m2->m[2].y + m1->m[2].w * m2->m[2].z + m1->m[3].w * m2->m[2].w;

	mat.m[3].x = m1->m[0].x * m2->m[3].x + m1->m[1].x * m2->m[3].y + m1->m[2].x * m2->m[3].z + m1->m[3].x * m2->m[3].w;
	mat.m[3].y = m1->m[0].y * m2->m[3].x + m1->m[1].y * m2->m[3].y + m1->m[2].y * m2->m[3].z + m1->m[3].y * m2->m[3].w;
	mat.m[3].z = m1->m[0].z * m2->m[3].x + m1->m[1].z * m2->m[3].y + m1->m[2].z * m2->m[3].z + m1->m[3].z * m2->m[3].w;
	mat.m[3].w = m1->m[0].w * m2->m[3].x + m1->m[1].w * m2->m[3].y + m1->m[2].w * m2->m[3].z + m1->m[3].w * m2->m[3].w;

	*out = mat;
}

void Matrix4MultiplyMatrix4Fast(MATRIX4 * out, MATRIX4 * m1, MATRIX4 * m2)
{
	out->m[0].x = m1->m[0].x * m2->m[0].x + m1->m[1].x * m2->m[0].y + m1->m[2].x * m2->m[0].z + m1->m[3].x * m2->m[0].w;
	out->m[0].y = m1->m[0].y * m2->m[0].x + m1->m[1].y * m2->m[0].y + m1->m[2].y * m2->m[0].z + m1->m[3].y * m2->m[0].w;
	out->m[0].z = m1->m[0].z * m2->m[0].x + m1->m[1].z * m2->m[0].y + m1->m[2].z * m2->m[0].z + m1->m[3].z * m2->m[0].w;
	out->m[0].w = m1->m[0].w * m2->m[0].x + m1->m[1].w * m2->m[0].y + m1->m[2].w * m2->m[0].z + m1->m[3].w * m2->m[0].w;

	out->m[1].x = m1->m[0].x * m2->m[1].x + m1->m[1].x * m2->m[1].y + m1->m[2].x * m2->m[1].z + m1->m[3].x * m2->m[1].w;
	out->m[1].y = m1->m[0].y * m2->m[1].x + m1->m[1].y * m2->m[1].y + m1->m[2].y * m2->m[1].z + m1->m[3].y * m2->m[1].w;
	out->m[1].z = m1->m[0].z * m2->m[1].x + m1->m[1].z * m2->m[1].y + m1->m[2].z * m2->m[1].z + m1->m[3].z * m2->m[1].w;
	out->m[1].w = m1->m[0].w * m2->m[1].x + m1->m[1].w * m2->m[1].y + m1->m[2].w * m2->m[1].z + m1->m[3].w * m2->m[1].w;

	out->m[2].x = m1->m[0].x * m2->m[2].x + m1->m[1].x * m2->m[2].y + m1->m[2].x * m2->m[2].z + m1->m[3].x * m2->m[2].w;
	out->m[2].y = m1->m[0].y * m2->m[2].x + m1->m[1].y * m2->m[2].y + m1->m[2].y * m2->m[2].z + m1->m[3].y * m2->m[2].w;
	out->m[2].z = m1->m[0].z * m2->m[2].x + m1->m[1].z * m2->m[2].y + m1->m[2].z * m2->m[2].z + m1->m[3].z * m2->m[2].w;
	out->m[2].w = m1->m[0].w * m2->m[2].x + m1->m[1].w * m2->m[2].y + m1->m[2].w * m2->m[2].z + m1->m[3].w * m2->m[2].w;

	out->m[3].x = m1->m[0].x * m2->m[3].x + m1->m[1].x * m2->m[3].y + m1->m[2].x * m2->m[3].z + m1->m[3].x * m2->m[3].w;
	out->m[3].y = m1->m[0].y * m2->m[3].x + m1->m[1].y * m2->m[3].y + m1->m[2].y * m2->m[3].z + m1->m[3].y * m2->m[3].w;
	out->m[3].z = m1->m[0].z * m2->m[3].x + m1->m[1].z * m2->m[3].y + m1->m[2].z * m2->m[3].z + m1->m[3].z * m2->m[3].w;
	out->m[3].w = m1->m[0].w * m2->m[3].x + m1->m[1].w * m2->m[3].y + m1->m[2].w * m2->m[3].z + m1->m[3].w * m2->m[3].w;
}

void Matrix4MultiplyMatrix4Mode2D(MATRIX4 * out, MATRIX4 * m1, MATRIX4 * m2, float& lookdis)
{
	out->m[0].x = m1->m[0].x * m2->m[0].x + m1->m[1].x * m2->m[0].y + m1->m[2].x * m2->m[0].z + m1->m[3].x * m2->m[0].w;
	out->m[0].y = m1->m[0].y * m2->m[0].x + m1->m[1].y * m2->m[0].y + m1->m[2].y * m2->m[0].z + m1->m[3].y * m2->m[0].w;
	out->m[0].z = m1->m[0].z * m2->m[0].x + m1->m[1].z * m2->m[0].y + m1->m[2].z * m2->m[0].z + m1->m[3].z * m2->m[0].w;

	out->m[1].x = m1->m[0].x * m2->m[1].x + m1->m[1].x * m2->m[1].y + m1->m[2].x * m2->m[1].z + m1->m[3].x * m2->m[1].w;
	out->m[1].y = m1->m[0].y * m2->m[1].x + m1->m[1].y * m2->m[1].y + m1->m[2].y * m2->m[1].z + m1->m[3].y * m2->m[1].w;
	out->m[1].z = m1->m[0].z * m2->m[1].x + m1->m[1].z * m2->m[1].y + m1->m[2].z * m2->m[1].z + m1->m[3].z * m2->m[1].w;

	out->m[2].x = m1->m[0].x * m2->m[2].x + m1->m[1].x * m2->m[2].y + m1->m[2].x * m2->m[2].z + m1->m[3].x * m2->m[2].w;
	out->m[2].y = m1->m[0].y * m2->m[2].x + m1->m[1].y * m2->m[2].y + m1->m[2].y * m2->m[2].z + m1->m[3].y * m2->m[2].w;
	out->m[2].z = m1->m[0].z * m2->m[2].x + m1->m[1].z * m2->m[2].y + m1->m[2].z * m2->m[2].z + m1->m[3].z * m2->m[2].w;

	out->m[3].x = m1->m[0].x * m2->m[3].x + m1->m[1].x * m2->m[3].y + m1->m[2].x * m2->m[3].z + m1->m[3].x * m2->m[3].w;
	out->m[3].y = m1->m[0].y * m2->m[3].x + m1->m[1].y * m2->m[3].y + m1->m[2].y * m2->m[3].z + m1->m[3].y * m2->m[3].w;
	out->m[3].z = m1->m[0].z * m2->m[3].x + m1->m[1].z * m2->m[3].y + m1->m[2].z * m2->m[3].z + m1->m[3].z * m2->m[3].w;

	out->m[0].w = 0.0f;
	out->m[1].w = 0.0f;
	out->m[2].w = 0.0f;
	out->m[3].w = lookdis;
}

void Matrix3MultiplyMatrix4Fast(MATRIX4 * out, MATRIX4 * m4, MATRIX3 * m)
{
	MATRIX4 m3;

	m3.m[0].x = m->m[0].x;
	m3.m[0].y = m->m[0].y;
	m3.m[0].z = m->m[0].z;
	
	m3.m[1].x = m->m[1].x;
	m3.m[1].y = m->m[1].y;
	m3.m[1].z = m->m[1].z;
	
	m3.m[2].x = m->m[2].x;
	m3.m[2].y = m->m[2].y;
	m3.m[2].z = m->m[2].z;

	m3.m[0].w = 0.0f;
	m3.m[1].w = 0.0f;
	m3.m[2].w = 0.0f;
	m3.m[3].w = 1.0f;
	m3.m[3].x = 0.0f;
	m3.m[3].y = 0.0f;
	m3.m[3].z = 0.0f;

	out->m[0].x = m4->m[0].x * m3.m[0].x + m4->m[1].x * m3.m[0].y + m4->m[2].x * m3.m[0].z + m4->m[3].x * m3.m[0].w;
	out->m[0].y = m4->m[0].y * m3.m[0].x + m4->m[1].y * m3.m[0].y + m4->m[2].y * m3.m[0].z + m4->m[3].y * m3.m[0].w;
	out->m[0].z = m4->m[0].z * m3.m[0].x + m4->m[1].z * m3.m[0].y + m4->m[2].z * m3.m[0].z + m4->m[3].z * m3.m[0].w;
	out->m[0].w = m4->m[0].w * m3.m[0].x + m4->m[1].w * m3.m[0].y + m4->m[2].w * m3.m[0].z + m4->m[3].w * m3.m[0].w;

	out->m[1].x = m4->m[0].x * m3.m[1].x + m4->m[1].x * m3.m[1].y + m4->m[2].x * m3.m[1].z + m4->m[3].x * m3.m[1].w;
	out->m[1].y = m4->m[0].y * m3.m[1].x + m4->m[1].y * m3.m[1].y + m4->m[2].y * m3.m[1].z + m4->m[3].y * m3.m[1].w;
	out->m[1].z = m4->m[0].z * m3.m[1].x + m4->m[1].z * m3.m[1].y + m4->m[2].z * m3.m[1].z + m4->m[3].z * m3.m[1].w;
	out->m[1].w = m4->m[0].w * m3.m[1].x + m4->m[1].w * m3.m[1].y + m4->m[2].w * m3.m[1].z + m4->m[3].w * m3.m[1].w;

	out->m[2].x = m4->m[0].x * m3.m[2].x + m4->m[1].x * m3.m[2].y + m4->m[2].x * m3.m[2].z + m4->m[3].x * m3.m[2].w;
	out->m[2].y = m4->m[0].y * m3.m[2].x + m4->m[1].y * m3.m[2].y + m4->m[2].y * m3.m[2].z + m4->m[3].y * m3.m[2].w;
	out->m[2].z = m4->m[0].z * m3.m[2].x + m4->m[1].z * m3.m[2].y + m4->m[2].z * m3.m[2].z + m4->m[3].z * m3.m[2].w;
	out->m[2].w = m4->m[0].w * m3.m[2].x + m4->m[1].w * m3.m[2].y + m4->m[2].w * m3.m[2].z + m4->m[3].w * m3.m[2].w;

	out->m[3].x = m4->m[0].x * m3.m[3].x + m4->m[1].x * m3.m[3].y + m4->m[2].x * m3.m[3].z + m4->m[3].x * m3.m[3].w;
	out->m[3].y = m4->m[0].y * m3.m[3].x + m4->m[1].y * m3.m[3].y + m4->m[2].y * m3.m[3].z + m4->m[3].y * m3.m[3].w;
	out->m[3].z = m4->m[0].z * m3.m[3].x + m4->m[1].z * m3.m[3].y + m4->m[2].z * m3.m[3].z + m4->m[3].z * m3.m[3].w;
	out->m[3].w = m4->m[0].w * m3.m[3].x + m4->m[1].w * m3.m[3].y + m4->m[2].w * m3.m[3].z + m4->m[3].w * m3.m[3].w;
}