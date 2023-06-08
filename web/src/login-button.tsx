import { useAuth0 } from '@auth0/auth0-react';
import Button from '@mui/material/Button';

export const LoginButton = () => {
	const { loginWithRedirect } = useAuth0();

	const handleLogin = async () => {
		await loginWithRedirect({
			appState: {
				returnTo: '/'
			}
		});
	};

	return (
		<Button variant='contained' className='button__login' onClick={handleLogin}>
			Log In
		</Button>
	);
};
