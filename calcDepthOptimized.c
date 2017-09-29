
// include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

// include OpenMP
#if !defined(_MSC_VER)
#include <pthread.h>
#endif
#include <omp.h>

#include "calcDepthOptimized.h"
#include "calcDepthNaive.h"

/* DO NOT CHANGE ANYTHING ABOVE THIS LINE. */

// float displacementNaive(int dx, int dy)
// {
// 	float squaredDisplacement = dx * dx + dy * dy;
// 	float displacement = sqrt(squaredDisplacement);
// 	return displacement;
// }

// void displacementOptimized(int dx, int dy)
// {
// 	float displacement;

// }

void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement)
{
	//calcDepthNaive(depth, left, right, imageWidth, imageHeight, featureWidth, featureHeight, maximumDisplacement);
	int x, y;
	memset(depth, 0, imageHeight * imageWidth * sizeof(float));
	#pragma omp parallel for private(x, y)
	/* The two outer for loops iterate through each pixel */
	for (y = featureHeight; y < imageHeight - featureHeight; y++)
	{
		// for (x = 0; x < imageWidth; x++)
		for (x = featureWidth; x < imageWidth - featureWidth; x++)
		{	
			/* Set the depth to 0 if looking at edge of the image where a feature box cannot fit. */

			float minimumSquaredDifference = -1;
			int minimumDy = 0;
			int minimumDx = 0;


			/* Iterate through all feature boxes that fit inside the maximum displacement box. 
			   centered around the current pixel. */


			for (int dy = -maximumDisplacement; dy <= maximumDisplacement; dy++)
			{
				for (int dx = -maximumDisplacement; dx <= maximumDisplacement; dx++)
				{
					/* Skip feature boxes that dont fit in the displacement box. */
					if (y + dy - featureHeight < 0 || y + dy + featureHeight >= imageHeight || x + dx - featureWidth < 0 || x + dx + featureWidth >= imageWidth)
					{
						continue;
					}

					float squaredDifference = 0;

					/* Sum the squared difference within a box of +/- featureHeight and +/- featureWidth. */
					int rowWidth = featureWidth * 2 + 1;
					int remain = rowWidth;
					__m128 rowdifference = _mm_setzero_ps();
					__m128 tempLeft, tempRight, diff;

					if ((minimumSquaredDifference < squaredDifference) && (minimumSquaredDifference != -1)) 
					{
						continue;
					}
					
					if (remain > 4)
					{
						for (int i = 0; i < rowWidth/4 * 4; i+= 4)
						{
							for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
							{

								int leftY = y + boxY;
								int rightY = y + dy + boxY;
								int leftX = x - featureWidth;
								int rightX = x + dx - featureWidth;

								tempLeft = _mm_loadu_ps(left + leftY * imageWidth + leftX + i);
								tempRight = _mm_loadu_ps(right + rightY * imageWidth + rightX + i);
								diff =_mm_sub_ps(tempLeft,tempRight);
								rowdifference = _mm_add_ps(_mm_mul_ps(diff, diff), rowdifference);
							}
						}
						squaredDifference += rowdifference[0] + rowdifference[1] + rowdifference[2] +rowdifference[3];
						remain = remain % 4;

					}
					if ((minimumSquaredDifference < squaredDifference) && (minimumSquaredDifference != -1)) 
					{
						continue;
					}
					for (int i = rowWidth/4 * 4; i < rowWidth; i++) 
					{
						for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)

						{
							int leftY = y + boxY;
							int rightY = y + dy + boxY;
							int leftX = x - featureWidth;
							int rightX = x + dx - featureWidth;
							squaredDifference += (left[leftY * imageWidth + leftX + i] - right[rightY * imageWidth + rightX + i]) * (left[leftY * imageWidth + leftX + i] - right[rightY * imageWidth + rightX + i]);
						}
					}
					
					/* 
					Check if you need to update minimum square difference. 
					This is when either it has not been set yet, the current
					squared displacement is equal to the min and but the new
					displacement is less, or the current squared difference
					is less than the min square difference.
					*/
					if ((minimumSquaredDifference == -1) || ((minimumSquaredDifference == squaredDifference) && (displacementNaive(dx, dy) < displacementNaive(minimumDx, minimumDy))) || (minimumSquaredDifference > squaredDifference))
					{
						minimumSquaredDifference = squaredDifference;
						minimumDx = dx;
						minimumDy = dy;
					}
				}
			}

			/* 
			Set the value in the depth map. 
			If max displacement is equal to 0, the depth value is just 0.
			*/
			if (minimumSquaredDifference != -1 && maximumDisplacement != 0)
			{
				depth[y * imageWidth + x] = displacementNaive(minimumDx, minimumDy);
				
			}
			
		}
	}

}
















































